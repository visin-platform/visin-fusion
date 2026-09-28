"""Visualization uploads include only rendered files and carry frame metadata."""

from unittest.mock import Mock

from visin_fusion.integrations.visualization_uploader import queue_visualizations


def test_queue_visualizations_uploads_existing_kinds(tmp_path, capsys):
    for kind in ('segment', 'overlay'):
        directory = tmp_path / kind
        directory.mkdir()
        (directory / 'frame.png').write_bytes(b'png')
    run = Mock()
    assert queue_visualizations(run, 4, 'epoch-id', str(tmp_path), 'frame.png') == 2
    assert run.upload_visualization.call_count == 2
    assert run.upload_visualization.call_args_list[0].args == (
        4, str(tmp_path / 'segment' / 'frame.png'), 'segment'
    )
    assert run.upload_visualization.call_args_list[1].kwargs == {
        'epoch_uuid': 'epoch-id',
        'metadata': {'image_name': 'frame.png', 'source': 'visualization_script'},
    }
    assert 'Skipping compare' in capsys.readouterr().out
