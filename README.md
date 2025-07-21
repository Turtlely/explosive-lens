# Explosive Lens Simulation
![image](https://github.com/Turtlely/explosive-lens/blob/35c43d232c9f097b058987cb74f7898e3fc7b425/display.gif)

Explosive lenses were used in the design of the first nuclear bombs to focus a divergent spherical detonation wave into a convergent spherical detonation wave via a combination of high-velocity and low-velocity explosives machined to have a certain geometry.

These devices are possible because shock waves follow Snell's law; thus, the problem can be treated like an optical lens design problem.

A handful of parameters, such as detonation velocities in both media, distance between the detonator and plutonium core, and time to detonation, give the actual geometry of the lens. 

The following equation defines the lens geometry:

$\frac{\sqrt{x^2+y^2}}{v_1}+ \frac{\sqrt{(d-x)^2+y^2}}{v_2} - t_d = 0$

Where:
- $d$: Distance between the source of the detonation wave and the center of the plutonium core to be crushed
- $v_1$: Detonation wave velocity in the fast explosive
- $v_2$: Detonation wave velocity in the slow explosive
- $t_d$: Total time between the detonation wave's start and the plutonium core's total collapse to a point.
