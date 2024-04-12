import sympy

def loop(n, given, input):
    currString = given
    for i in range(n):
        currString = currString.replace("x", input)
    currString += "-x"
    
    expr = sympy.sympify(currString)
    print(expr)
    equation = sympy.Eq(expr, 0)
    return equation
        
equation = loop(1, "(7/2)*x*(1-x)", "(7/2)*x*(1-x)")
solutions = sympy.solve(equation, 'x')

# Evaluate and round each solution to the nearest hundredth
print(solutions)
rounded_solutions = [round(sympy.N(solution), 4) for solution in solutions]
print(rounded_solutions)
