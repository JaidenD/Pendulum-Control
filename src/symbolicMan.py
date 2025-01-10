import sympy as sp
# time coordinate
t = sp.Symbol('t')

# position coordinates
x = sp.Function('x')(t)
θ1 = sp.Function('θ1')(t)
θ2 = sp.Function('θ2')(t)

# params
l1, l2, m1, m2, m3, g = sp.symbols('l1 l2 m1 m2 m3 g', positive=True)

# first derivatives
dx = x.diff(t)
dθ1 = θ1.diff(t)
dθ2 = θ2.diff(t)

### convention: height coordinate is positive in direction of pendulum
p1 = sp.Matrix([x,0])
p2 = sp.Matrix([x + l1 * sp.sin(θ1), l1 * sp.cos(θ1)])
p3 = sp.Matrix([x + l1 * sp.sin(θ1) + l2 * sp.sin(θ2), l1 * sp.cos(θ1) + l2 * sp.cos(θ2)])

#############################################################################
#                                   v p1 mass = m1
#                        ________________________   < x
#                                    \
#                                     \
#                                      \
#                                       o < p2 mass = m2, angle = θ1
#                                       |
#                                       |
#                                       |
#                                       o < p3 mass = m3, angle = θ2
#
#############################################################################


# potential
U = (m1 + m2)*l1*g*p2[1] + m1*l2*g*p3[1]

# compute kinetic energy

dp1 = p1.diff(t)
dp2 = p2.diff(t)
dp3 = p3.diff(t)

T = 1/2*(m1+m2+m3)*dp1.dot(dp1) + 1/2*(m2+m3)*dp3.dot(dp2) + 1/2*(m3)*dp3.dot(dp3)

# Lagrangian
L = T-U

print("\n")
print("L = ")
print(L)

# Compute equations of motion

