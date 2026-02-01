import numpy as np

def MDD(series):
    pivot = 0
    min = 0
    drawdown = float("inf") #O certo é iniciar o drawdown com +∞, assim sempre pega o primeiro valor real:

    for i in range(len(series)):

        if series[i] > series[pivot]:
            pivot = i

        min_test = i + np.argmin(series[i:])
        test = series[min_test]/series[pivot] - 1
        if drawdown > test:
            drawdown = test
            min = min_test

    return [pivot, min, drawdown]

if __name__ == "__main__":

    series = np.random.randint(100, 200, size=20)
    print(series)

    pivot, min_idx, dd = MDD(series)

    print(f"pivot index = {pivot}")
    print(f"pivot value = {series[pivot]}")
    print(f"min index = {min_idx}")
    print(f"min value = {series[min_idx]}")
    print(f"drawdown = {dd * 100:.2f}%")