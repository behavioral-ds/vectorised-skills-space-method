class MatrixSubsetIndexes:
    indexes: list[int]

    def __init__(
        self,
        matrix_subset_slice: tuple[int, int] | list[int],
    ):
        if type(matrix_subset_slice) is tuple:
            self.indexes = [
                i for i in range(matrix_subset_slice[0], matrix_subset_slice[1])
            ]
        else:
            self.indexes = matrix_subset_slice

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        for index in self.indexes:
            yield index
