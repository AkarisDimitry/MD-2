/* VelVerlet.c */

float first_step(float **x, float **v, float **a, float **xf, float **vf, float dt, int n) {
	
	int i; 

	for (i = 0; i < n; i++) {
		xf[i][0] = x[i][0] + v[i][0]*dt + 0.5*a[i][0]*dt*dt;
		vf[i][0] = v[i][0] + 0.5*a[i][0]*dt;
		
		xf[i][1] = x[i][1] + v[i][1]*dt + 0.5*a[i][1]*dt*dt;
		vf[i][1] = v[i][1] + 0.5*a[i][1]*dt;

		xf[i][2] = x[i][2] + v[i][2]*dt + 0.5*a[i][2]*dt*dt;
		vf[i][2] = v[i][2] + 0.5*a[i][2]*dt;
	}
	return 0;
}




