#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing engine for model evaluation.
"""
import time
import torch
import numpy as np
from tqdm import tqdm
from utils.helpers import relabel_annotation
from models.registry import segment
from utils.metrics import compute_ap_for_class, store_predictions_for_ap


class TestingEngine:
    """Handles model testing and evaluation."""
    
    def __init__(self, model, metrics_calculator, config, device):
        self.model = model
        self.metrics_calc = metrics_calculator
        self.config = config
        self.device = device
    
    def test(self, dataloader, modality, num_classes):
        """Run testing on a dataloader and return results."""
        self.model.eval()
        
        # Initialize accumulators for IoU/precision/recall
        accumulators = self.metrics_calc.create_accumulators(self.device)
        
        # Initialize storage for AP calculation (pixel-wise predictions and targets)
        all_predictions = {cls: [] for cls in self.metrics_calc.eval_classes}
        all_targets = {cls: [] for cls in self.metrics_calc.eval_classes}
        
        # Track inference time
        total_inference_time = 0.0
        total_samples = 0
        total_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader)
            for i, batch in enumerate(progress_bar):
                # Move data to device
                rgb = batch['rgb'].to(self.device, non_blocking=True)
                lidar = batch['lidar'].to(self.device, non_blocking=True)
                anno = batch['anno'].to(self.device, non_blocking=True)
                
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Time the forward pass
                inference_start = time.time()
                
                # Forward pass: the segmentation map, whatever the model (models/registry.py)
                output_seg = segment(self.model, self.config, rgb, lidar)
                
                # Synchronize and record time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                total_samples += rgb.size(0)
                total_batches += 1
                
                # Relabel annotation
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)
                
                # Update accumulators for IoU/precision/recall
                batch_overlap, batch_pred, batch_label, batch_union = self.metrics_calc.update_accumulators(
                    accumulators, output_seg, anno, num_classes
                )
                
                # Store predictions and targets for AP calculation
                store_predictions_for_ap(output_seg, anno, all_predictions, all_targets,
                                         self.metrics_calc.eval_classes, self.metrics_calc.eval_indices)
                
                # Calculate batch metrics for progress bar
                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                array_indices = self._get_array_indices()
                
                # Update progress bar
                progress_desc = ' '.join([
                    f'{cls.upper()}:IoU->{batch_IoU[array_indices[j]]:.4f}'
                    for j, cls in enumerate(self.metrics_calc.eval_classes)
                ])
                progress_bar.set_description(progress_desc)
        
        # Compute final metrics including proper AP
        results = self._compute_final_results(accumulators, all_predictions, all_targets)
        
        # Print results
        self._print_results(results)
        
        # Return results and inference stats
        inference_stats = {
            'total_inference_time': total_inference_time,
            'total_samples': total_samples,
            'total_batches': total_batches
        }
        
        return results, inference_stats
    
    def _get_array_indices(self):
        """Positions of the eval classes in the per-class metric arrays (which exclude background)."""
        return [idx - 1 for idx in self.metrics_calc.eval_indices]
    
    def _compute_final_results(self, accumulators, all_predictions, all_targets):
        """Compute final test results with proper AP calculation."""
        cum_IoU = accumulators['overlap'] / accumulators['union']
        cum_precision = accumulators['overlap'] / accumulators['pred']
        cum_recall = accumulators['overlap'] / accumulators['label']
        
        # Filter to eval classes
        array_indices = self._get_array_indices()
        eval_IoU = cum_IoU[array_indices]
        eval_precision = cum_precision[array_indices]
        eval_recall = cum_recall[array_indices]
        
        # Calculate F1 and AP
        results = {}
        for i, cls in enumerate(self.metrics_calc.eval_classes):
            iou = self.metrics_calc.sanitize_value(eval_IoU[i].item())
            precision = self.metrics_calc.sanitize_value(eval_precision[i].item())
            recall = self.metrics_calc.sanitize_value(eval_recall[i].item())
            
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1 = self.metrics_calc.sanitize_value(f1)
            
            # Calculate proper AP using stored predictions
            ap = compute_ap_for_class(cls, all_predictions, all_targets)
            ap = self.metrics_calc.sanitize_value(ap)
            
            results[cls] = {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'ap': ap
            }
        
        # Calculate additional overall metrics
        pixel_accuracy = accumulators['pixel_correct'] / accumulators['pixel_total'] if accumulators['pixel_total'] > 0 else 0.0
        mean_accuracy = torch.mean(eval_recall).item()  # Mean of per-class recalls
        fw_iou = 0.0
        if accumulators['class_pixels'].sum() > 0:
            weights = accumulators['class_pixels'] / accumulators['class_pixels'].sum()
            fw_iou = (weights * eval_IoU).sum().item()
        
        # Add overall metrics
        confusion_matrix_labels = [cls['name'] for cls in sorted(self.config['Dataset']['train_classes'], key=lambda x: x['index'])]
        class_results = [results[cls] for cls in self.metrics_calc.eval_classes]
        results['overall'] = {
            'mIoU_foreground': torch.mean(eval_IoU).item(),
            'mean_precision': float(np.mean([r['precision'] for r in class_results])),
            'mean_recall': float(np.mean([r['recall'] for r in class_results])),
            'mean_f1': float(np.mean([r['f1_score'] for r in class_results])),
            'mean_ap': float(np.mean([r['ap'] for r in class_results])),
            'mean_accuracy': mean_accuracy,
            'fw_iou': fw_iou,
            'pixel_accuracy': float(pixel_accuracy),
            'confusion_matrix': accumulators['confusion_matrix'].cpu().tolist(),
            'confusion_matrix_labels': confusion_matrix_labels
        }
        
        return results
    
    def _print_results(self, results):
        """Print test results."""
        print('-----------------------------------------')
        for cls, metrics in results.items():
            if cls == 'overall':
                continue
            print(f'{cls.upper()}: IoU->{metrics["iou"]:.4f} '
                  f'Precision->{metrics["precision"]:.4f} '
                  f'Recall->{metrics["recall"]:.4f} '
                  f'F1->{metrics["f1_score"]:.4f} '
                  f'AP->{metrics["ap"]:.4f}')
        
        # Print overall metrics
        overall = results.get('overall', {})
        if overall:
            print('-----------------------------------------')
            print(f'mIoU (foreground): {overall.get("mIoU_foreground", 0):.4f}')
            print(f'Mean Accuracy: {overall.get("mean_accuracy", 0):.4f}')
            print(f'Frequency-Weighted IoU: {overall.get("fw_iou", 0):.4f}')
            print(f'Pixel Accuracy: {overall.get("pixel_accuracy", 0):.4f}')
        print('-----------------------------------------')
