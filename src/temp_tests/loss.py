from functions.loss import *

if __name__ == "__main__":
    w1 = np.zeros((2, 2), dtype=np.double)
    x = np.array([[1, 2],
                  [2, 3],
                  [0, 0],
                  [-1, -1]], dtype=np.double)
    y = np.array([[1, 2],
                  [2, 3],
                  [0, 0],
                  [-1, -1]], dtype=np.double)
    loss = Huber()
    for i in range(1, 100000):
        g1 = loss.gradient(x, y, w1)

        if i % 1000 == 0:
            print(loss(x, y, w1))

        w1 -= 10**-3*g1

    print(w1)