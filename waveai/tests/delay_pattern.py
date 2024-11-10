import torch

from waveai.utils.pattern import DelayPattern


def test_delay_pattern_mono():
    """Test the mono delay pattern"""

    pattern = DelayPattern(stereo=False)

    input = torch.tensor(
        [
            [
                [3, 2, 4, 8, 7, 2, 4, 5, 9, 6],
                [0, 9, 9, 4, 8, 3, 5, 0, 4, 2],
                [0, 5, 8, 4, 3, 6, 0, 8, 3, 9],
                [0, 5, 2, 7, 4, 5, 3, 2, 4, 8],
            ]
        ]
    )

    out, _ = pattern.build_delay_pattern_mask(input, -10, 100)

    output_tgt = torch.tensor(
        [
            [
                [3, 2, 4, 8, 7, 2, 4, 5, 9, 6],
                [-10, 0, 9, 9, 4, 8, 3, 5, 0, 4],
                [-10, -10, 0, 5, 8, 4, 3, 6, 0, 8],
                [-10, -10, -10, 0, 5, 2, 7, 4, 5, 3],
            ]
        ]
    )

    assert torch.all(out == output_tgt), f"Expected {output_tgt}, got {out}"


def test_delay_pattern_stereo():
    """Test the stereo delay pattern"""

    pattern = DelayPattern(stereo=True)

    input = torch.tensor(
        [
            [
                [3, 2, 4, 8, 7, 2, 4, 5, 9, 6],
                [0, 9, 9, 4, 8, 3, 5, 0, 4, 2],
                [0, 5, 8, 4, 3, 6, 0, 8, 3, 9],
                [0, 5, 2, 7, 4, 5, 3, 2, 4, 8],
            ]
        ]
    )

    out, _ = pattern.build_delay_pattern_mask(input, -10, 100)

    output_tgt = torch.tensor(
        [
            [
                [3, 2, 4, 8, 7, 2, 4, 5, 9, 6],
                [0, 5, 8, 4, 3, 6, 0, 8, 3, 9],
                [-10, 0, 9, 9, 4, 8, 3, 5, 0, 4],
                [-10, 0, 5, 2, 7, 4, 5, 3, 2, 4],
            ]
        ]
    )

    assert torch.all(out == output_tgt), f"Expected {output_tgt}, got {out}"
