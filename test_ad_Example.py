import taichi as ti
ti.init(debug=True)

x = ti.field(dtype=ti.f32, shape=6, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
x.fill(1.0)


@ti.func
def func2():
    for j in range(x.shape[0]):
        ti.atomic_add(loss[None], j * ti.sin(x[j]))

@ti.kernel
def compute_loss():
    func2()

with ti.ad.Tape(loss=loss, validation=True):
    compute_loss()

print('dy/dx_0 =', x.grad, ' at x_0 =', x)