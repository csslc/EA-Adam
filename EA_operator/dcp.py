
# decompose a multi-objective optimization problem into a single-objective optimization problems
# for gradient-based optimization

def dcp_obj(f1, f2, type):
    if type == 'WS': # Weighted  sum  approach
        lam1 = f1/(f1+f2)
        lam2 = f2/(f1+f2)
        f = lam1 * f1 + lam2 * f2

    return f