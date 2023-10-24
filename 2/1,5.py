
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    m_a = len(A[0])
    m_b = len(b)
    if m_a != m_b:
        raise ValueError()

   # TODO: Perform gaussian elimination

    #check if any 0 on diagonal, if yes, then no pivoting
    if any(np.diag(A) == 0):
        raise ValueError("Zero division error!")

    #Gauß without pivoting
    for row in range(0, m_a-1):
        for i in range(row+1, m_a):
            f = A[i, row] / A[row, row]
            for j in range(row, m_a):
                A[i, j] = A[i, j] - f * A[row, j]
            b[i] = b[i] - f * b[row]


    #Gaus with pivoting
    for i in range(min(m_a, m_a)):  # for each column on the diagonal
        if (A[i][i] == 0):  # Find a non-zero pivot and swap rows
            coloumn = [A[j][i] for k in range(i, m_a)]
            pivot = coloumn.index(max(coloumn))
            var = A[i]
            A[i] = A[pivot]
            A[pivot] = var
            if (A[i][i] == 0):
                raise ValueError('A hat eine Spalte von 0')
        for k in range(i + 1, m_a):
            ratio = A[k][i] / A[i][i]  # Ratio of (i,j) elt by (j,j) (diagonal) elt
            A[k] = [A[k][j] - ratio * A[i][j] for j in range(m_a)]

        return A, b

    '''
    #Gaus with pivoting
    for i in range(m_a-1):
        max_value = abs(A[i:, i]).argmax() + i
        if A[max_value, i] == 0:
            raise ValueError ('No pivoting! Matrix is singular!')
        if max_value != i:
            A[[i, max_value]] = A[[max_value, i]]
            b[[i, max_value]] = b[[max_value, i]]
        for row in range(i+1, m_a):
            temp = A[row][i]/A[i][i]
            A[row][i] = temp
            for col in range(i + 1, m_a):
                A[row][col] = A[row][col] - temp * A[i][col]
                # Equation solution column
            b[row] = b[row] - temp * b[i]
    '''
