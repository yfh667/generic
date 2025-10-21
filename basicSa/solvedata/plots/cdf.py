import matplotlib.pyplot as plt
import numpy as np


def plot_cdf(intervals):
    """
    Plot Cumulative Distribution Function (CDF) of time intervals
    :param intervals: List of time intervals in seconds
    """
    # Data filtering
    valid_intervals = [x for x in intervals if x > 0]
    if len(valid_intervals) < 1:
        print("No valid data to plot CDF")
        return

    # Sort data
    sorted_intervals = np.sort(valid_intervals)
    n = len(sorted_intervals)

    # Calculate CDF
    cdf = np.arange(1, n + 1) / n

    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.step(sorted_intervals, cdf, where='post', label='CDF', linewidth=2)

    # Add percentile markers
    median = np.median(sorted_intervals)
    p90 = np.percentile(sorted_intervals, 90)

    plt.axvline(median, color='red', linestyle='--',
                label=f'Median ({median:.1f}s)')
    plt.axvline(p90, color='green', linestyle=':',
                label=f'90th percentile ({p90:.1f}s)')

    # Formatting
    plt.xlabel('Time Interval (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Path Change Intervals')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, sorted_intervals[-1] * 1.1)
    plt.show()

#
# # 在 plot 函数中调用示例
# qujian = [14, 16, 1, 79, 11, 3]  # 示例数据
# plot_cdf(qujian)
