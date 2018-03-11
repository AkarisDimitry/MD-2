/* VelVerlet.c */

void pair_force(float *s1, float *s2, float *f1, float *f2, int rcut) {
	float r2, r8, r14;
	float fg1, fg2, fg3;

        r2 = (s1[0]-s2[0])*(s1[0]-s2[0]) + (s1[1]-s2[1])*(s1[1]-s2[1]) + (s1[2]-s2[2])*(s1[2]-s2[2]);
	
	if (r2 < rcut*rcut) { 
	    r8 = r2*r2*r2*r2;
	    r14 = r8*r2*r2*r2;
	    
	    fg1 = (48/r14 - 24/r8)*(s1[0]-s2[0]);
            fg2 = (48/r14 - 24/r8)*(s1[1]-s2[1]);
	    fg3 = (48/r14 - 24/r8)*(s1[2]-s2[2]);

	    f1[0] = f1[0] + fg1;
	    f1[1] = f1[1] + fg2;
       	    f1[2] = f1[2] + fg3;
            
	    f2[0] = f2[0] - fg1;
	    f2[1] = f2[1] - fg2;
	    f2[2] = f2[2] - fg3;




	}	
}

void pair_energ( float *s1, float *s2, float *E, float rcut )  {
    float r2, r6, r12;
    r2 = (s1[0]-s2[0])*(s1[0]-s2[0]) + (s1[1]-s2[1])*(s1[1]-s2[1]) + (s1[2]-s2[2])*(s1[2]-s2[2]);
    r6 = r2*r2*r2;
    r12 = r6*r6;
    if ( r2 < rcut*rcut) {
	E[0] = E[0] + 4/r12 - 4/r6;
    }
}

void forces(float *x, float *v, float *F, float *E, int n, int rcut) {
    int i, j;
    float Eshift, rcut6;
    rcut6 = rcut*rcut*rcut*rcut*rcut*rcut;
    Eshift = 4/rcut6 - 4/rcut6*rcut6;

    for (i = 0; i < n; i++) {
	for (j = i + 1; j < n; j++) {
	   
	    pair_force( x+3*i, x+3*j, F+3*i, F+3*j, rcut );
   	    pair_energ( x+3*i, x+3*j, E, rcut); 
	}
    }
}





