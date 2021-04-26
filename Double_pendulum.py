import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

m1 = 1
m2 = 1
l1 = 1
l2 = 1.5
mu = (m2/m1)
lam = l2/l1
g = 9.8

dt = 0.001
T = 500
t = np.arange(0,T,dt)
tau = np.sqrt(g/l1)*t

M = np.zeros((2,2))

theta1 = np.zeros(tau.shape)
theta2 = np.zeros(tau.shape)
dtheta1 = np.zeros(tau.shape)
dtheta2 = np.zeros(tau.shape)

theta1[0] = np.pi/4
theta2[0] = np.pi/4
dtheta1[0] = 0
dtheta2[0] = 0

def diff_solv_dtheta(theta,dtheta):
    
    theta1 = theta[0]
    theta2 = theta[1]
    dtheta1 = dtheta[0]
    dtheta2 = dtheta[1]
    
    M = np.zeros((2,2))
    M[0,0] = 1+mu
    M[0,1] = mu*lam*np.cos(theta1-theta2)
    M[1,0] = mu*lam*np.cos(theta1-theta2)
    M[1,1] = mu*lam*lam
    
    f = np.zeros((2,1))
    f[0] = (-(1+mu)*np.sin(theta1)) - (mu*lam*dtheta2*dtheta2*np.sin(theta1-theta2))
    f[1] = (-mu*lam*np.sin(theta2)) + (mu*lam*dtheta1*dtheta1*np.sin(theta1-theta2))
    
    inv_M = np.linalg.inv(M)
    
    y = np.dot(inv_M,f)
    
    return y

for i in range(len(tau)-1):
    theta = np.array([theta1[i],theta2[i]]).reshape(2,1)
    dtheta = np.array([dtheta1[i],dtheta2[i]]).reshape(2,1)
    
    k1 = diff_solv_dtheta(theta,dtheta)
    k2 = diff_solv_dtheta(theta,(dtheta+dt*0.5*k1))
    k3 = diff_solv_dtheta(theta,(dtheta+dt*0.5*k2))
    k4 = diff_solv_dtheta(theta,(dtheta+dt*k3))
    
    dtheta1[i+1] = dtheta1[i] + 1/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt
    dtheta2[i+1] = dtheta2[i] + 1/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt    
    theta1[i+1] = theta1[i] + dt*dtheta1[i]
    theta2[i+1] = theta2[i] + dt*dtheta2[i]
    
   

x1 = np.sin(theta1)
x2 = x1 + np.sin(theta2)
y1 = -np.cos(theta1)
y2 = y1 - np.cos(theta2)

L = l1+l2+0.5

fig = plt.figure()
plt.style.use('dark_background')
ax = plt.axes(xlim = (-L,L),ylim = (-L,L))
line1, = ax.plot([],[],lw = 2,color = 'white')
line2, = ax.plot([],[],lw = 2,color = 'white')
line3, = ax.plot([],[],lw = 1,color='cyan',alpha = 0.5)


def init():
    line1.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    return line1,line2,line3

def animate(k):
    i = 50*k
    line1.set_data([0,x1[i]],[0,y1[i]])
    bob1 = plt.Circle((x1[i],y1[i]),0.1,fc = 'r')
    ax.add_patch(bob1)
    line2.set_data([x1[i],x2[i]],[y1[i],y2[i]])
    bob2 = plt.Circle((x2[i],y2[i]),0.1,fc = 'r')
    ax.add_patch(bob2)
    line3.set_data(x2[:i],y2[:i])
    
    return line1,line2,line3,bob1,bob2

ani = animation.FuncAnimation(fig, animate,init_func = init, frames = len(tau), interval = 1,blit=True)
plt.show()
#ani.save("DoublePendulum.gif",writer = 'imagemagick',fps = 60)