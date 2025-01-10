import sympy as sp
# time coordinate
t = sp.Symbol('t')

# position coordinates
x = sp.Function('x')(t)
θ1 = sp.Function('θ1')(t)
θ2 = sp.Function('θ2')(t)

# params
l1, l2, M, m1, m2, g = sp.symbols('l1 l2 M m1 m2 g', positive=True)

# first derivatives
dx = x.diff(t)
dθ1 = θ1.diff(t)
dθ2 = θ2.diff(t)

Q = sp.Matrix([x,0])
q1 = sp.Matrix([x + l1 * sp.sin(θ1), l1 * sp.cos(θ1)])
q2 = sp.Matrix([x + l1 * sp.sin(θ1) + l2 * sp.sin(θ2), l1 * sp.cos(θ1) + l2 * sp.cos(θ2)])

### convention: height coordinate is positive in direction of pendulum
###             angles are taken from the verticle axis


#############################################################################
#                                    
#                        ________________________ < position = Q, mass = M
#                                    \
#                                     \
#                                      \
#                                       o < position = q1, mass = m1, angle = θ1
#                                       |
#                                       |
#                                       |
#                                       o < position = q1, mass = m1, angle = θ2
#
#############################################################################


# potential
U = (m1 + m2)*l1*g*q1[1] + M*l2*g*q2[1]

# compute kinetic energy

dQ = Q.diff(t)
dq1 = q1.diff(t)
dq2 = q2.diff(t)

T = 1/2*(M+m1+m2)*dQ.dot(dQ) + 1/2*(m1+m2)*dq2.dot(dq1) + 1/2*(m2)*dq2.dot(dq2)

# Lagrangian
L = T-U

print("\n")
print("L = ")
print(L)

# Compute equations of motion

X = sp.Matrix([x, θ1, θ2])
dX = sp.Matrix([dx, dθ1, dθ2])

# Compute derivatives
L_dX = [L.diff(dXi) for dXi in dX]  # ∂L/∂(dq_i)
L_X = [L.diff(Xi) for Xi in X]      # ∂L/∂q_i

# Time derivatives of ∂L/∂(dq_i)
dL_dX_dt = [sp.diff(L_dXi, t) for L_dXi in L_dX]

# Euler-Lagrange equations
eom = [dL_dX_dt[i] - L_X[i] for i in range(len(X))]

# Display equations of motion
for i, eq in enumerate(eom):
    print(f"Equation for {X[i]}:")
    print(eq.simplify())