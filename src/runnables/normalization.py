# from src.preconditioning.normalization import *
#
# if __name__ == '__main__':
#     normalizer = GaussianNormalizer()
#
#     data = np.array([[727.7,  727.7],
#                      [1086.5, 1086.5],
#                      [1091.0, 1091.0],
#                      [1361.3, 1361.3],
#                      [1490.5, 1490.5],
#                      [1956.1, 1956.1]])
#     print(normalizer(data))
#     normalizer = MinMaxNormalizer()
#
#     data = np.array([[727.7,  727.7],
#                      [1086.5, 1086.5],
#                      [1091.0, 1091.0],
#                      [1361.3, 1361.3],
#                      [1490.5, 1490.5],
#                      [1956.1, 1956.1]])
#     print(normalizer(data))
#
#     normalizer = DecimalScaling()
#
#     data = np.array([[727.7,  727.7],
#                      [1086.5, 1086.5],
#                      [1091.0, 1091.0],
#                      [1361.3, 1361.3],
#                      [1490.5, 1490.5],
#                      [1956.1, 1956.1]])
#     print(normalizer(data))
#
#     normalizer = GaussianOutlierRemoval()
#
#     data = np.array([[727.7,  727.7],
#                      [1086.5, 1086.5],
#                      [1091.0, 1091.0],
#                      [10000000.3, 1361.3],
#                      [1490.5, 1490.5],
#                      [1956.1, 10000000]])
#     print(normalizer(data))
