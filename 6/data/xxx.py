# TODO: calculate gradient and area before changing the surface
# gradient = np.zeros_like(v)
gradient = surface_area_gradient(v, f)
area = 0.0
area = surface_area(v, f)

# TODO: calculate indices of vertices whose position can be changed
indices = []
for i in range(len(f)):
    if not np.isin(f[i, 0] or f[i, 1] or f[i, 2], c):
        indices.append(f[i])
indices = np.asarray(indices)

# TODO: find suitable step size so that area can be decreased, don't change v yet
step = 1.0

v_copy = v.copy()
new_area = surface_area(v_copy, f)
while area - new_area <= epsilon:
    for i in indices:
        v_copy[i] += gradient[i] * step
    new_area = surface_area(v_copy, f)
    step /= 2
    v_copy = v.copy()
    if step <= 10 * epsilon:
        break

# TODO: now update vertex positions in v
for j in indices:
    v[j] += gradient[j] * step
new_area = surface_area(v, f)

# TODO: Check if new area differs only epsilon from old area
# Return (True, area, v, gradient) to show that we converged and otherwise (False, area, v, gradient)
if np.abs(area - new_area) > epsilon:
    return False, area, v, gradient
return True, area, v, gradient

indices = []
for i in range(len(v)):
    if i not in c:
        indices.append(i)