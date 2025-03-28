"""Tests for ribs.visualize.

For image comparison tests, read these instructions:
https://matplotlib.org/3.3.1/devel/testing.html#writing-an-image-comparison-test.
Essentially, after running a new test for the first time in the _root_ directory
of this repo, copy the image output from result_images/visualize_test to
tests/extras/baseline_images/visualize_test. For instance, for
``test_cvt_archive_heatmap_with_samples``, run::

    cp result_images/visualize_test/cvt_archive_heatmap_with_samples.png \
        tests/baseline_images/visualize_test/

Assuming the output is as expected (and assuming the code is deterministic), the
test should now pass when it is re-run.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive, GridArchive, SlidingBoundariesArchive
from ribs.visualize import (cvt_archive_heatmap, grid_archive_heatmap,
                            parallel_axes_plot,
                            sliding_boundaries_archive_heatmap)

# pylint: disable = redefined-outer-name


@pytest.fixture(autouse=True)
def clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


def add_uniform_sphere(archive, x_range, y_range):
    """Adds points from the negative sphere function in a 100x100 grid.

    The solutions are the same as the BCs (the (x,y) coordinates).

    x_range and y_range are tuples of (lower_bound, upper_bound).
    """
    for x in np.linspace(x_range[0], x_range[1], 100):
        for y in np.linspace(y_range[0], y_range[1], 100):
            archive.add(
                solution=np.array([x, y]),
                objective_value=-(x**2 + y**2),  # Negative sphere.
                behavior_values=np.array([x, y]),
            )


def add_uniform_3d_sphere(archive, x_range, y_range, z_range):
    """Adds points from the negative sphere function in a 100x100x100 grid.

    The solutions are the same as the BCs (the (x,y,z) coordinates).

    x_range, y_range, and z_range are tuples of (lower_bound, upper_bound).
    """
    for x in np.linspace(x_range[0], x_range[1], 40):
        for y in np.linspace(y_range[0], y_range[1], 40):
            for z in np.linspace(z_range[0], z_range[1], 40):
                archive.add(
                    solution=np.array([x, y, z]),
                    objective_value=-(x**2 + y**2 + z**2),  # Negative sphere.
                    behavior_values=np.array([x, y, z]),
                )


def add_random_sphere(archive, x_range, y_range):
    """Adds 1000 random points from the negative sphere function.

    Solutions, BCs, and ranges are same as in add_uniform_sphere.
    """
    # Use random BCs to make the boundaries shift.
    rng = np.random.default_rng(10)
    for _ in range(1000):
        x, y = rng.uniform((x_range[0], y_range[0]), (x_range[1], y_range[1]))
        archive.add(
            solution=np.array([x, y]),
            objective_value=-(x**2 + y**2),
            behavior_values=np.array([x, y]),
        )


#
# Archive fixtures.
#


@pytest.fixture(scope="module")
def grid_archive():
    """Deterministically created GridArchive."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.
    archive = GridArchive([10, 10], [(-1, 1), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    add_uniform_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def long_grid_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = GridArchive([10, 10], [(-2, 2), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    add_uniform_sphere(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def three_d_grid_archive():
    """Deterministic archive, but there are three behavior axes of different
    sizes, and some of the axes are not totally filled."""
    archive = GridArchive([10, 10, 10], [(-2, 2), (-1, 1), (-2, 1)], seed=42)
    archive.initialize(solution_dim=3)
    add_uniform_3d_sphere(archive, (0, 2), (-1, 1), (-1, 0))
    return archive


@pytest.fixture(scope="module")
def cvt_archive():
    """Deterministically created CVTArchive."""
    archive = CVTArchive(100, [(-1, 1), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    archive.initialize(solution_dim=2)
    add_uniform_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def long_cvt_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = CVTArchive(100, [(-2, 2), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    archive.initialize(solution_dim=2)
    add_uniform_sphere(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def sliding_archive():
    """Deterministically created SlidingBoundariesArchive."""
    archive = SlidingBoundariesArchive([10, 20], [(-1, 1), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    add_random_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def long_sliding_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = SlidingBoundariesArchive([10, 20], [(-2, 2), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    add_random_sphere(archive, (-2, 2), (-1, 1))
    return archive


#
# Tests for all heatmap functions. Unfortunately, these tests are hard to
# parametrize because the image_comparison decorator needs the filename, and
# pytest does not seem to pass params to decorators. It is important to keep
# these tests separate so that if the test fails, we can immediately retrieve
# the correct result image. For instance, if we have 3 tests in a row and the
# first one fails, no result images will be generated for the remaining two
# tests.
#


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding"])
def test_heatmap_fails_on_non_2d(archive_type):
    archive = {
        "grid":
            lambda: GridArchive([20, 20, 20], [(-1, 1)] * 3),
        "cvt":
            lambda: CVTArchive(100, [(-1, 1)] * 3, samples=100),
        "sliding":
            lambda: SlidingBoundariesArchive([20, 20, 20], [(-1, 1)] * 3),
    }[archive_type]()
    archive.initialize(solution_dim=2)  # Arbitrary.

    with pytest.raises(ValueError):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
        }[archive_type](archive)


@image_comparison(baseline_images=["grid_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive)


@image_comparison(baseline_images=["sliding_boundaries_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive)


@image_comparison(baseline_images=["grid_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_custom_axis__grid(grid_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, ax=ax)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_custom_axis__cvt(cvt_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, ax=ax)


@image_comparison(baseline_images=["sliding_boundaries_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_custom_axis__sliding(sliding_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, ax=ax)


@image_comparison(baseline_images=["grid_archive_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long__grid(long_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(long_grid_archive)


@image_comparison(baseline_images=["cvt_archive_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long__cvt(long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(long_cvt_archive)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long__sliding(long_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(long_sliding_archive)


@image_comparison(baseline_images=["grid_archive_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_square__grid(long_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(long_grid_archive, square=True)


@image_comparison(baseline_images=["cvt_archive_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_square__cvt(long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(long_cvt_archive, square=True)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_square__sliding(long_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(long_sliding_archive, square=True)


@image_comparison(baseline_images=["grid_archive_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_transpose__grid(long_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(long_grid_archive, transpose_bcs=True)


@image_comparison(baseline_images=["cvt_archive_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_transpose__cvt(long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(long_cvt_archive, transpose_bcs=True)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_transpose__sliding(long_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(long_sliding_archive, transpose_bcs=True)


@image_comparison(baseline_images=["grid_archive_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_limits__grid(grid_archive):
    # Negative sphere function should have range (-2, 0). These limits should
    # give a more uniform-looking archive.
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_limits__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_limits__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["grid_archive_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_listed_cmap__grid(grid_archive):
    # cmap consists of primary red, green, and blue.
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["cvt_archive_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_listed_cmap__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_listed_cmap"],
    remove_text=False,
    extensions=["png"])
def test_heatmap_listed_cmap__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive,
                                       cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["grid_archive_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_coolwarm_cmap__grid(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, cmap="coolwarm")


@image_comparison(baseline_images=["cvt_archive_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_coolwarm_cmap__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, cmap="coolwarm")


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_coolwarm_cmap"],
    remove_text=False,
    extensions=["png"])
def test_heatmap_coolwarm_cmap__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, cmap="coolwarm")


#
# Miscellaneous heatmap tests.
#


@image_comparison(baseline_images=["grid_archive_heatmap_with_boundaries"],
                  remove_text=False,
                  extensions=["png"])
def test_grid_archive_with_boundaries(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive,
                         pcm_kwargs={
                             "edgecolor": "black",
                             "linewidth": 0.1
                         })


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_boundaries"],
    remove_text=False,
    extensions=["png"])
def test_sliding_archive_with_boundaries(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, boundary_lw=0.5)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_samples"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_samples(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, plot_samples=True)


#
# Parallel coordinate plot test
#


@image_comparison(baseline_images=["parallel_axes_2d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_2d(grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive)


@image_comparison(baseline_images=["parallel_axes_3d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive)


@image_comparison(baseline_images=["parallel_axes_3d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_ax(three_d_grid_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, ax=ax)


@image_comparison(baseline_images=["parallel_axes_3d_custom_order"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_order(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, bc_order=[1, 2, 0])


@image_comparison(baseline_images=["parallel_axes_3d_custom_names"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_names(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive,
                       bc_order=[(1, 'One'), (2, 'Two'), (0, 'Zero')])


@image_comparison(baseline_images=["parallel_axes_3d_coolwarm"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_coolwarm_cmap(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, cmap='coolwarm')


@image_comparison(baseline_images=["parallel_axes_3d_width2_alpha2"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_width2_alpha2(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, linewidth=2.0, alpha=0.2)


@image_comparison(baseline_images=["parallel_axes_3d_custom_objective_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_objective_limits(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, vmin=-2.0, vmax=-1.0)


@image_comparison(baseline_images=["parallel_axes_3d_sorted"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_sorted(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, sort_archive=True)


@image_comparison(baseline_images=["parallel_axes_3d_vertical_cbar"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_vertical_cbar(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, cbar_orientation='vertical')
