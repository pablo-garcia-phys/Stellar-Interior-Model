import numpy as np
import matplotlib.pyplot as plt

class StellarModel:
    def __init__(self, M_tot, R_tot, L_tot, T_c, X, Y):
        """
        Initialize the stellar model.
        """
        # 1. Input Parameters 
        self.M_tot = M_tot
        self.R_tot = R_tot
        self.L_tot = L_tot
        self.T_c_ini = T_c
        self.X = X
        self.Y = Y
        self.Z = 1 - X - Y

        # 2. Mean Molecular Weight (mu) 
        self.mu = 1 / (2*self.X + 0.75*self.Y + 0.5*self.Z)
        
        # 3. Constants for the Equations 
        
        # Mass constant
        self.C_m = 0.01523 * self.mu
        
        # Pressure constant 
        self.C_p = 8.084 * self.mu
        
        # Temperature constant (Radiative) 
        self.C_t = 0.01679 * self.Z * (1+self.X) * self.mu**2  
        
        # Temperature constant (Convective) 
        self.C_t_conv = 3.234 * self.mu 

        print(f"Model initialized. Mu: {self.mu:.4f}")

    def calculate_density(self, P, T):
        """
        Calculates density (rho) in g/cm^3 given P and T in model units.
        """
        # Converting model units to cgs
        P_cgs = P * 1e15
        T_cgs = T * 1e7
        R_gas = 8.31446e7 
        
        rho = (P_cgs * self.mu) / (R_gas * T_cgs)
        return rho
    
    def calculate_opacity(self, rho, T):
        """
        Calculates opacity (kappa) using Kramer's law.
        Units: cm^2/g
        """
        C_k = 4.34e25 
        T_cgs = T * 1e7
        kappa = C_k * self.Z * (1 + self.X) * rho * (T_cgs**-3.5)
        return kappa

    #  ENERGY GENERATION METHODS 
    def _lookup_pp_coeffs(self, T):
        """Look up table for Proton-Proton chain."""
        T6 = T / 1e6 # Correcting the units
        T_table = T * 10.0 
        
        if 4.0 <= T_table < 6.0:   return -6.84, 6.0
        elif 6.0 <= T_table < 9.5: return -6.04, 5.0
        elif 9.5 <= T_table < 12.0: return -5.56, 4.5
        elif 12.0 <= T_table < 16.5: return -5.02, 4.0
        elif 16.5 <= T_table <= 24.0: return -4.40, 3.5
        else: return None, None

    def _lookup_cn_coeffs(self, T):
        """Look up table for CNO cycle."""
        T_table = T * 10.0 
        
        if 12.0 <= T_table < 16.0:   return -22.2, 20
        elif 16.0 <= T_table < 22.5: return -19.8, 18
        elif 22.5 <= T_table < 27.5: return -17.1, 16
        elif 27.5 <= T_table < 36.0: return -15.6, 15
        elif 36.0 <= T_table <= 50.0: return -12.5, 13
        else: return None, None

    def generate_energy(self, T, rho):
        """
        Calculates epsilon (erg/g/s) choosing max(PP, CN). 
        """
        # 1. Proton-Proton
        log_e1_pp, nu_pp = self._lookup_pp_coeffs(T)
        if log_e1_pp is not None:
            eps_1 = 10**log_e1_pp
            # For PP: X1=X, X2=X
            T_term = (T * 10.0) ** nu_pp
            eps_pp = eps_1 * self.X * self.X * rho * T_term
        else:
            eps_pp = 0.0

        # 2. CNO Cycle
        log_e1_cn, nu_cn = self._lookup_cn_coeffs(T)
        if log_e1_cn is not None:
            eps_1 = 10**log_e1_cn
            # For CNO: X1=X, X2=Z/3 
            T_term = (T * 10.0) ** nu_cn
            eps_cn = eps_1 * self.X * (self.Z/3.0) * rho * T_term
        else:
            eps_cn = 0.0

        return max(eps_pp, eps_cn)

    def radiative_gradients(self, r, P, T, M, L):
        """
        Calculates the gradients (derivatives) assuming RADIATIVE transport.
        Returns: (dP_dr, dT_dr, dM_dr, dL_dr)
        """
        rho = self.calculate_density(P, T)
        epsilon = self.generate_energy(T, rho)
        
        # EQUATIONS 
        
        # dM/dr 
        dM_dr = self.C_m * (P / T) * (r**2)
        
        # dP/dr 
        dP_dr = -self.C_p * (P / T) * (M / r**2)
        
        # dL/dr 
        dL_dr = 0.01256637 * (r**2) * rho * epsilon
        
        # dT/dr 
        dT_dr = -self.C_t * (P**2 / T**8.5) * (L / r**2)
        
        return dP_dr, dT_dr, dM_dr, dL_dr
    
    def solve_starting_layers(self):
        """
        Calculates the first 3 layers (0, 1, 2) using analytic approximations.
        Returns a list of dictionaries with the layer data.
        """
        # 1. Define integration parameters 
        r_start = 0.9 * self.R_tot
        # h is negative because we integrate inwards
        h = - r_start / 100.0  
        
        # 2. Calculate constants A1 and A2 
        # A1 = 1.9022 * mu * M_tot
        A1 = 1.9022 * self.mu * self.M_tot
        
        # A2 = 10.645 * sqrt( M_tot / (mu * Z * (1+X) * L_tot) )
        term_inside_sqrt = self.M_tot / (self.mu * self.Z * (1 + self.X) * self.L_tot)
        A2 = 10.645 * np.sqrt(term_inside_sqrt)
        
        # 3. Calculate first 3 layers
        layers = []
        
        print("-" * 60)
        print(f"{'Phase'} {'i':<3} {'r':<10} {'P':<12} {'T':<12} {'L':<10} {'M':<10}")
        print("-" * 60)

        for i in range(3):
            # Current radius
            r_i = r_start + i * h
            
            # Analytic T
            # T = A1 * (1/r - 1/R_tot)
            T_i = A1 * (1.0/r_i - 1.0/self.R_tot)
            
            # Analytic P 
            # P = A2 * T^4.25
            P_i = A2 * (T_i**4.25)
            
            # Assumptions for the outer layers:
            M_i = self.M_tot
            L_i = self.L_tot
            
            # Calculate auxiliary values
            rho_i = self.calculate_density(P_i, T_i)
            # We assume radiative transport in the surface
            nabla_rad = "N/A" # We'll calculate this later
            
            # Store data
            layer_data = {
                'i': i,
                'r': r_i,
                'P': P_i,
                'T': T_i,
                'M': M_i,
                'L': L_i,
                'rho': rho_i
            }
            layers.append(layer_data)
            
            # Print to check against 
            print(f"START {i:<3} {r_i:.5f}  {P_i:.7f}  {T_i:.7f}  {L_i:.6f}  {M_i:.6f}  ")
            
        return layers, h # We return h because we'll need it for the next steps
    
    def _get_derivatives(self, r, P, T, M, L):
        """Wrapper to get derivatives as a numpy array."""
        dP, dT, dM, dL = self.radiative_gradients(r, P, T, M, L)
        return np.array([dP, dT, dM, dL])
    
    def _get_energy_label(self, T, rho):
        """Helper to return 'PP', 'CN' or '--' string based on dominant cycle."""
        if T < 1.0: return "--" # Temperature too low for significant fusion (< 10^7 K)
        
        # Calculate actual epsilon values
        eps_pp_val = 0
        eps_cn_val = 0
        
        # PP Chain
        log_e1_pp, nu_pp = self._lookup_pp_coeffs(T)
        if log_e1_pp is not None:
            eps_1 = 10**log_e1_pp
            T_term = (T * 10.0) ** nu_pp
            eps_pp_val = eps_1 * self.X * self.X * rho * T_term
            
        # CNO Cycle
        log_e1_cn, nu_cn = self._lookup_cn_coeffs(T)
        if log_e1_cn is not None:
            eps_1 = 10**log_e1_cn
            T_term = (T * 10.0) ** nu_cn
            eps_cn_val = eps_1 * self.X * (self.Z/3.0) * rho * T_term
            
        total_eps = max(eps_pp_val, eps_cn_val)
        
        # Threshold to display generation (approx 10^-20 or just relative to L)
        if total_eps < 1e-15: return "--" 
        
        if eps_pp_val >= eps_cn_val: return "PP"
        return "CN"
    
    def solve_phase_A11(self, layers, h):
        """
        Executes Phase A.1.1 (Layers 3, 4, and 5) forcing M=M_tot and L=L_tot.
        """
        
        print("\n" + "="*85)
        print(f"{'E':<4} {'phase':<8} {'i':<4} {'r':<10} {'P':<12} {'T':<12} {'L':<10} {'M':<10}")
        print("="*85)
        
        # 1. Retrieve historical gradients from layers 0, 1, and 2
        # At startup, f_M and f_L are 0.
        grads = []
        for layer in layers:
            dP, dT, _, _ = self.radiative_gradients(layer['r'], layer['P'], layer['T'], self.M_tot, self.L_tot)
            # Force 0.0 in derivatives of M and L for startup
            grads.append(np.array([dP, dT, 0.0, 0.0]))
            
            # Print START layers with requested format
            print(f"{'--':<4} {'START':<8} {layer['i']:<4} {layer['r']:.5f}  {layer['P']:.7f}  {layer['T']:.7f}  {self.L_tot:.6f}  {self.M_tot:.6f}")

        # 2. Loop for layers 3, 4, and 5
        i = 2 
        while i < 5: 
            # Preparation (Step 1) 
            r_i = layers[i]['r']
            v_i = np.array([layers[i]['P'], layers[i]['T'], layers[i]['M'], layers[i]['L']])
            f_i = grads[i]
            
            # Predictor
            f_im1 = grads[i-1] 
            f_im2 = grads[i-2]
            
            d1_i = h * (f_i - f_im1)
            d2_i = h * (f_i - 2*f_im1 + f_im2)
            
            v_est = v_i + h*f_i + 0.5*d1_i + (5/12)*d2_i
            
            # Initial estimated variables
            P_est = v_est[0]
            T_est = v_est[1]
            # M_est and L_est are ignored, we will use constants M_tot and L_tot
            
            # Corrector Loop 
            r_next = r_i + h
            converged = False
            tol = 0.0001
            
            # Final iteration variables
            P_cal = 0.0
            T_cal = 0.0
            
            while not converged:
                # Step 4: Calculate dP/dr using M_TOT (constant)
                dP_next = -self.C_p * (P_est / T_est) * (self.M_tot / r_next**2)
                
                # Pressure Corrector
                d1_next_P = h * (dP_next - f_i[0])
                P_cal = v_i[0] + h*dP_next - 0.5*d1_next_P
                
                # Step 5: Check P convergence
                if abs(P_cal - P_est)/(P_cal + 1e-30) > tol:
                    P_est = P_cal
                    continue 
                
                # Step 7: Calculate dT/dr using L_TOT (constant)
                dT_next = -self.C_t * (P_cal**2 / T_est**8.5) * (self.L_tot / r_next**2)
                
                # Temperature Corrector
                d1_next_T = h * (dT_next - f_i[1])
                T_cal = v_i[1] + h*dT_next - 0.5*d1_next_T
                
                # Step 8: Check T convergence
                if abs(T_cal - T_est)/(T_cal + 1e-30) > tol:
                    T_est = T_cal
                    continue 
                
                converged = True
            
            # Step 3: Calculate mass derivative for history
            # Although we force M=M_tot in the table, we need to calculate real dM/dr 
            # so that, when entering phase A.1.2, the predictor has data.
            dM_next = self.C_m * (P_cal / T_cal) * (r_next**2)
            
            # Luminosity remains strictly constant (dL/dr = 0)
            dL_next = 0.0
            
            # Save gradients
            new_grad = np.array([dP_next, dT_next, dM_next, dL_next])
            grads.append(new_grad)
            
            # Save and display the layer forcing M = M_tot
            new_layer = {
                'i': i+1, 
                'r': r_next, 
                'P': P_cal, 
                'T': T_cal, 
                'M': self.M_tot, # <--- KEY POINT: Force the constant
                'L': self.L_tot
            }
            layers.append(new_layer)
            
            # Print exact row
            print(f"{'--':<4} {'A.1.1':<8} {i+1:<4} {r_next:.5f}  {P_cal:.7f}  {T_cal:.7f}  {self.L_tot:.6f}  {self.M_tot:.6f}")
            
            i += 1
            
        return layers
    
    def solve_phase_A12(self, layers, h):
        """
        Executes Phase A.1.2: Radiative envelope, variable mass, constant luminosity.
        """
        print("-" * 85)
        # Informational header
        print("Starting Phase A.1.2 (Variable M, Constant L)...")
        
        # Retrieve historical gradients from previous layers
        # We need to recalculate them because in A.1.1 we forced dM/dr = 0, 
        # but now we need the real mass gradient to start the predictor.
        grads = []
        for layer in layers:
            dP, dT, dM, dL = self.radiative_gradients(layer['r'], layer['P'], layer['T'], layer['M'], layer['L'])
            # In A.1.2, L remains constant, so we force dL = 0
            grads.append(np.array([dP, dT, dM, 0.0]))

        # Start where the previous phase left off
        i = len(layers) - 1 
        
        phase_running = True
        
        while phase_running:
            # PREPARATION (Step 1)
            r_i = layers[i]['r']
            v_i = np.array([layers[i]['P'], layers[i]['T'], layers[i]['M'], layers[i]['L']])
            f_i = grads[i]
            
            # Predictor (Adams-Bashforth 2nd order)
            f_im1 = grads[i-1] 
            f_im2 = grads[i-2]
            
            d1_i = h * (f_i - f_im1)
            d2_i = h * (f_i - 2*f_im1 + f_im2)
            
            v_est = v_i + h*f_i + 0.5*d1_i + (5/12)*d2_i
            
            # Initial estimated variables
            P_est, T_est = v_est[0], v_est[1]
            M_est = v_est[2]
            # L_est is not used, we use L_tot
            
            # CORRECTOR LOOP 
            r_next = r_i + h
            converged = False
            tol = 0.0001
            
            # Final iteration variables
            P_cal, T_cal, M_cal = 0.0, 0.0, 0.0
            
            while not converged:
                # Step 3: Calculate dM/dr using ESTIMATED values
                dM_next = self.C_m * (P_est / T_est) * (r_next**2)
                d1_next_M = h * (dM_next - f_i[2])
                M_cal = v_i[2] + h*dM_next - 0.5*d1_next_M
                
                # Step 4: Calculate dP/dr using newly calculated M_CAL
                dP_next = -self.C_p * (P_est / T_est) * (M_cal / r_next**2)
                d1_next_P = h * (dP_next - f_i[0])
                P_cal = v_i[0] + h*dP_next - 0.5*d1_next_P
                
                # Step 5: Check P convergence
                if abs(P_cal - P_est)/(P_cal + 1e-30) > tol:
                    P_est = P_cal
                    continue 

                # Step 7: Calculate dT/dr using L_TOT (constant)
                dT_next = -self.C_t * (P_cal**2 / T_est**8.5) * (self.L_tot / r_next**2)
                d1_next_T = h * (dT_next - f_i[1])
                T_cal = v_i[1] + h*dT_next - 0.5*d1_next_T
                
                # Step 8: Check T convergence
                if abs(T_cal - T_est)/(T_cal + 1e-30) > tol:
                    T_est = T_cal
                    continue 
                
                converged = True
            
            # Step 6: Verify if Luminosity remains constant 
            # Calculate what dL/dr WOULD be if there was energy generation
            rho_cal = self.calculate_density(P_cal, T_cal)
            epsilon = self.generate_energy(T_cal, rho_cal) 
            
            # Real theoretical dL 
            dL_next_theoretical = 0.01256637 * (r_next**2) * rho_cal * epsilon
            
            # Predict the new L
            d1_next_L = h * (dL_next_theoretical - f_i[3])
            # Corrector for L (although in A.1.2 we assume constant L for P and T, 
            # we need to calculate real L to see when to switch phases)

            L_cal = v_i[3] + h*dL_next_theoretical - 0.5*d1_next_L - (1/12)*0 # 2nd order term omitted for simplicity in check
            
            # PHASE CHANGE CRITERION 
            # Is L_cal approx L_tot?
            # The script maintains A.1.2 until the difference is appreciable.
            # We will use a strict but sufficient tolerance to reach layer 60.
            if abs(L_cal - self.L_tot)/self.L_tot > 0.0001: # 0.01% error
                print(f"--> Phase change detected at layer {i+1}. L begins to vary.")
                phase_running = False
                break
            
            # If we continue in A.1.2, force L = L_tot for the table and the next step
            L_final = self.L_tot
            dL_final = 0.0 # Force gradient 0 to keep the phase pure
            
            # Save gradients (Mass now varies, L does not)
            new_grad = np.array([dP_next, dT_next, dM_next, dL_final])
            grads.append(new_grad)
            
            # Check energy to print "PP", "CN" or "--"
            energy_str = self._get_energy_label(T_cal, rho_cal)
            
            # Save layer
            new_layer = {
                'i': i+1, 'r': r_next, 'P': P_cal, 'T': T_cal, 'M': M_cal, 'L': L_final
            }
            layers.append(new_layer)
            
            # Print 
            print(f"{energy_str:<4} {'A.1.2.':<8} {i+1:<4} {r_next:.5f}  {P_cal:.7f}  {T_cal:.7f}  {L_final:.6f}  {M_cal:.6f}")
            
            i += 1
            
            # Safety: Avoid infinite loop if something fails
            if i > 200: 
                print("Safety: Too many layers.")
                break
                
        return layers
    
    def solve_phase_A13(self, layers, h):
        """
        Executes Phase A.1.3: Radiative envelope, ALL variables (M, L, P, T) change.
        Stops immediately BEFORE adding the layer where convection is detected.
        """
        print("-" * 95)
        print(f"Starting Phase A.1.3 (Variable L and M). checking for convection...")
        print(f"{'E':<4} {'Phase':<8} {'i':<4} {'r':<10} {'P':<12} {'T':<12} {'L':<10} {'M':<10} {'n+1'}")
        print("-" * 95)
        
        # 1. Recalculate gradients history
        grads = []
        for layer in layers:
            dP, dT, dM, dL = self.radiative_gradients(layer['r'], layer['P'], layer['T'], layer['M'], layer['L'])
            grads.append(np.array([dP, dT, dM, dL]))

        # Start from the last layer calculated
        i = len(layers) - 1 
        convection_reached = False
        
        while not convection_reached:
            # --- PREPARATION ---
            r_i = layers[i]['r']
            v_i = np.array([layers[i]['P'], layers[i]['T'], layers[i]['M'], layers[i]['L']])
            f_i = grads[i]
            
            f_im1 = grads[i-1] 
            f_im2 = grads[i-2]
            
            # Predictor
            d1_i = h * (f_i - f_im1)
            d2_i = h * (f_i - 2*f_im1 + f_im2)
            v_est = v_i + h*f_i + 0.5*d1_i + (5/12)*d2_i
            P_est, T_est = v_est[0], v_est[1]
            
            # CORRECTOR LOOP 
            r_next = r_i + h
            converged = False
            tol = 0.0001
            
            P_cal, T_cal, M_cal, L_cal = 0.0, 0.0, 0.0, 0.0
            
            while not converged:
                # Step 3: Mass
                dM_next = self.C_m * (P_est / T_est) * (r_next**2)
                d1_next_M = h * (dM_next - f_i[2])
                M_cal = v_i[2] + h*dM_next - 0.5*d1_next_M
                
                # Step 4: Pressure
                dP_next = -self.C_p * (P_est / T_est) * (M_cal / r_next**2)
                d1_next_P = h * (dP_next - f_i[0])
                P_cal = v_i[0] + h*dP_next - 0.5*d1_next_P
                
                if abs(P_cal - P_est)/(P_cal + 1e-30) > tol:
                    P_est = P_cal
                    continue 

                # Step 6: Luminosity
                rho_est = self.calculate_density(P_cal, T_est)
                eps_est = self.generate_energy(T_est, rho_est)
                dL_next = 0.01256637 * (r_next**2) * rho_est * eps_est
                d1_next_L = h * (dL_next - f_i[3])
                L_cal = v_i[3] + h*dL_next - 0.5*d1_next_L

                # Step 7: Temperature
                dT_next = -self.C_t * (P_cal**2 / T_est**8.5) * (L_cal / r_next**2)
                d1_next_T = h * (dT_next - f_i[1])
                T_cal = v_i[1] + h*dT_next - 0.5*d1_next_T
                
                if abs(T_cal - T_est)/(T_cal + 1e-30) > tol:
                    T_est = T_cal
                    continue 
                
                converged = True
            
            # STEP 9: Check Convection (n+1) 
            if abs(dT_next) < 1e-20: 
                n_plus_1 = 99.9 
            else:
                n_plus_1 = (T_cal / P_cal) * (dP_next / dT_next)
            
            # Check condition BEFORE saving 
            if n_plus_1 <= 2.5:
                print(f"--> Convection detected at layer {i+1} (r={r_next:.5f}) with n+1={n_plus_1:.3f}")
                
                # Calculate K' here because we will need it for the next phase (A.2)
                # K' = P / T^2.5
                K_prime = P_cal / (T_cal**2.5)
                self.K_prime_val = K_prime # Store it in the class for later use
                print(f"--> Estimated Polytropic Constant K' = {K_prime:.6f}")
                print("--> Stopping Radiative Integration. This layer will be recalculated in Phase A.2.")
                
                convection_reached = True
                break # We exit the loop WITHOUT appending this layer to 'layers'
            
            # If valid (Radiative), we save and print
            grads.append(np.array([dP_next, dT_next, dM_next, dL_next]))
            
            rho_final = self.calculate_density(P_cal, T_cal)
            energy_str = self._get_energy_label(T_cal, rho_final)

            print(f"{energy_str:<4} {'A.1.3.':<8} {i+1:<4} {r_next:.5f}  {P_cal:.7f}  {T_cal:.7f}  {L_cal:.6f}  {M_cal:.6f}  {n_plus_1:.3f}")
            
            new_layer = {
                'i': i+1, 'r': r_next, 'P': P_cal, 'T': T_cal, 'M': M_cal, 'L': L_cal, 'n+1': n_plus_1
            }
            layers.append(new_layer)
            
            # Safety checks
            if r_next <= 0 or M_cal < 0:
                print("Error: Center reached or negative mass.")
                break
                
            i += 1
            
        return layers
    
    def solve_phase_A2(self, layers, h):
        """
        Executes Phase A.2: Convective Core.
        Uses the Polytropic relation P = K' * T^2.5.
        Stops when the center is reached (r <= 0).
        """
        print("-" * 95)
        print(f"Starting Phase A.2 (Convective Core). K' = {self.K_prime_val:.6f}")
        print(f"{'E':<4} {'Phase':<8} {'i':<4} {'r':<10} {'P':<12} {'T':<12} {'L':<10} {'M':<10} {'n+1'}")
        print("-" * 95)
        
        # 1. Recover history. We need gradients to start the Adams-Bashforth predictor.
        grads = []
        for layer in layers:
            dP, dT, dM, dL = self.radiative_gradients(layer['r'], layer['P'], layer['T'], layer['M'], layer['L'])
            grads.append(np.array([dP, dT, dM, dL]))

        # Start from the last valid layer (layer 81 in the test model)
        i = len(layers) - 1 
        center_reached = False
        
        while not center_reached:
            # PREPARATION 
            r_i = layers[i]['r']
            v_i = np.array([layers[i]['P'], layers[i]['T'], layers[i]['M'], layers[i]['L']])
            f_i = grads[i]
            
            f_im1 = grads[i-1] 
            f_im2 = grads[i-2]
            
            # Predictor differences
            d1_i = h * (f_i - f_im1)
            d2_i = h * (f_i - 2*f_im1 + f_im2)
            
            # PREDICTOR (Step 2bis) 
            # We only predict Temperature first, because P depends on T directly via Polytrope
            T_est = v_i[1] + h*f_i[1] + 0.5*d1_i[1] + (5/12)*d2_i[1]
            
            # Polytropic Relation: P = K' * T^2.5
            P_est = self.K_prime_val * (T_est**2.5)
            
            # CORRECTOR LOOP 
            r_next = r_i + h
            
            # Handle the center (r=0) strictly to avoid division by zero
            if r_next < 0.001: 
                r_next = 0.0
            
            converged = False
            tol = 0.0001
            
            P_cal, T_cal, M_cal, L_cal = 0.0, 0.0, 0.0, 0.0
            
            while not converged:
                # Step 3: Calculate Mass (using P_est, T_est)
                # Formula: dM = C_m * (P/T) * r^2
                if r_next == 0:
                    dM_next = 0.0 # Mass gradient is 0 at center
                else:
                    dM_next = self.C_m * (P_est / T_est) * (r_next**2)
                
                d1_next_M = h * (dM_next - f_i[2])
                M_cal = v_i[2] + h*dM_next - 0.5*d1_next_M
                
                # Step 7bis: Calculate Temperature Gradient
                # Formula (Convective): dT/dr = -C_t_conv * M / r^2
                # Note: C_t_conv = 3.234 * mu 
                
                if r_next == 0:
                    dT_next = 0.0 # Temperature gradient is 0 at center
                    T_cal = T_est # We accept the estimate
                else:
                    dT_next = -self.C_t_conv * (M_cal / (r_next**2))
                    d1_next_T = h * (dT_next - f_i[1])
                    T_cal = v_i[1] + h*dT_next - 0.5*d1_next_T
                
                # Step 8: Check Convergence for T
                if abs(T_cal - T_est)/(T_cal + 1e-30) > tol:
                    T_est = T_cal
                    # Re-calculate P based on new T
                    P_est = self.K_prime_val * (T_est**2.5)
                    continue 
                
                # Convergence achieved
                converged = True
                # Final P calculation
                P_cal = self.K_prime_val * (T_cal**2.5)

            # Step 6: Calculate Luminosity 
            
            rho_cal = self.calculate_density(P_cal, T_cal)
            eps_cal = self.generate_energy(T_cal, rho_cal)
            
            dL_next = 0.01256637 * (r_next**2) * rho_cal * eps_cal
            d1_next_L = h * (dL_next - f_i[3])
            L_cal = v_i[3] + h*dL_next - 0.5*d1_next_L - (1/12)*0 # Simplified corrector
            
            # Store Gradients (dP is needed for next steps history) 
            # We need dP/dr for the history of next steps, even if we calculate P via Polytrope
            if r_next == 0:
                dP_next = 0.0
            else:
                # dP/dr (Convective): -C_p * K' * T^1.5 * M / r^2
                # Or just generic hydrostatic: -C_p * (P/T) * M/r^2
                dP_next = -self.C_p * (P_cal / T_cal) * (M_cal / (r_next**2))

            grads.append(np.array([dP_next, dT_next, dM_next, dL_next]))
            
            # Determine Energy Label
            energy_str = self._get_energy_label(T_cal, rho_cal)
            
            # Print Row
            # n+1 is meaningless here (adiabatic by definition), usually not printed or set to 2.5
            print(f"{energy_str:<4} {'CONVEC':<8} {i+1:<4} {r_next:.5f}  {P_cal:.7f}  {T_cal:.7f}  {L_cal:.6f}  {M_cal:.6f}")
            
            new_layer = {
                'i': i+1, 'r': r_next, 'P': P_cal, 'T': T_cal, 'M': M_cal, 'L': L_cal
            }
            layers.append(new_layer)
            
            # STOP CONDITION
            if r_next <= 0:
                print("--> CENTER REACHED! Integration from surface complete.")
                center_reached = True
            
            # Fallback stop (negative mass due to numerical error near zero)
            '''if M_cal < 0:
                print("--> Negative mass detected near center. Stopping.")
                center_reached = True'''

            i += 1
            
        return layers
    
    def solve_starting_center_layers(self, T_c_guess, h_step):
        """
        Calculates the first 3 layers (0, 1, 2) starting from the CENTER (r=0)
        using Taylor series expansions
        """
        print("\n" + "="*95)
        print(f"Starting Phase B (Center Outwards) with Tc = {T_c_guess}")
        print(f"{'E':<4} {'Phase':<8} {'i':<4} {'r':<10} {'P':<12} {'T':<12} {'L':<10} {'M':<10}")
        print("="*95)
        
        layers_out = []
        
        # 1. Calculate Central values (r=0)
        # P_c = K' * T_c^2.5
        P_c = self.K_prime_val * (T_c_guess**2.5)
        
        # We need epsilon at the center for the L equation
        # Calculate rho_c first
        rho_c = self.calculate_density(P_c, T_c_guess)
        eps_c = self.generate_energy(T_c_guess, rho_c) # Calculates real epsilon
        
        # 2. Iterate for i = 0, 1, 2
        # Note: h_step passed here should be positive here (approx 0.1035)
        
        for i in range(3):
            r_i = i * h_step
            
            if r_i == 0:
                # Central Boundary Conditions
                P_i = P_c
                T_i = T_c_guess
                M_i = 0.0
                L_i = 0.0
                rho_i = rho_c
            else:
                # Expansion formulas 
                
                # Mass: M = 0.005077 * mu * K' * Tc^1.5 * r^3
                M_i = 0.005077 * self.mu * self.K_prime_val * (T_c_guess**1.5) * (r_i**3)
                
                # Luminosity: L = M * epsilon_c 
                L_i = M_i * eps_c
                
                # Temperature: T = Tc - 0.008207 * mu^2 * K' * Tc^1.5 * r^2
                T_i = T_c_guess - 0.008207 * (self.mu**2) * self.K_prime_val * (T_c_guess**1.5) * (r_i**2)
                
                # Pressure: P = K' * T^2.5
                P_i = self.K_prime_val * (T_i**2.5)
                
                # Recalculate rho
                rho_i = self.calculate_density(P_i, T_i)

            # Energy label
            energy_str = self._get_energy_label(T_i, rho_i)
            
            print(f"{energy_str:<4} {'CENTER':<8} {i:<4} {r_i:.5f}  {P_i:.7f}  {T_i:.7f}  {L_i:.6f}  {M_i:.6f}")
            
            layer_data = {
                'i': i, 'r': r_i, 'P': P_i, 'T': T_i, 'M': M_i, 'L': L_i
            }
            layers_out.append(layer_data)
            
        return layers_out
    
    def solve_core_outwards(self, layers_out, h_step, r_target):
        """
        Continues integration from center outwards using Phase A.2 algorithm
        until r >= r_target (the matching point).
        """
        # We need history gradients. Since we used Taylor, we must estimate them for the start.
        # Or simpler: calculate gradients for the 3 analytic layers we just made.
        grads = []
        for layer in layers_out:
            # We use radiative_gradients function but strictly it's convective logic
            # Since P is polytropic, dP/dr is fixed.
            # Let's just calculate the physics based gradients:
            if layer['r'] == 0:
                 grads.append(np.array([0.0, 0.0, 0.0, 0.0]))
            else:
                 dP, dT, dM, dL = self.radiative_gradients(layer['r'], layer['P'], layer['T'], layer['M'], layer['L'])
                 # Just need initial gradients for the Predictor.
                 # For Convection: dT/dr = -C_t_conv * M / r^2
                 dT_conv = -self.C_t_conv * (layer['M'] / layer['r']**2)
                 grads.append(np.array([dP, dT_conv, dM, dL]))

        i = 2
        reached_target = False
        
        while not reached_target:
            # SAME LOGIC AS PHASE A.2 (but h is positive) 
            r_i = layers_out[i]['r']
            v_i = np.array([layers_out[i]['P'], layers_out[i]['T'], layers_out[i]['M'], layers_out[i]['L']])
            f_i = grads[i]
            
            f_im1 = grads[i-1] 
            f_im2 = grads[i-2]
            
            d1_i = h_step * (f_i - f_im1)
            d2_i = h_step * (f_i - 2*f_im1 + f_im2)
            
            # Predict T
            T_est = v_i[1] + h_step*f_i[1] + 0.5*d1_i[1] + (5/12)*d2_i[1]
            P_est = self.K_prime_val * (T_est**2.5)
            
            # Corrector Loop
            r_next = r_i + h_step
            converged = False
            tol = 0.0001
            P_cal, T_cal, M_cal, L_cal = 0.0, 0.0, 0.0, 0.0
            
            while not converged:
                # Mass
                dM_next = self.C_m * (P_est / T_est) * (r_next**2)
                d1_next_M = h_step * (dM_next - f_i[2])
                M_cal = v_i[2] + h_step*dM_next - 0.5*d1_next_M
                
                # Temperature (Convective)
                dT_next = -self.C_t_conv * (M_cal / (r_next**2))
                d1_next_T = h_step * (dT_next - f_i[1])
                T_cal = v_i[1] + h_step*dT_next - 0.5*d1_next_T
                
                if abs(T_cal - T_est)/(T_cal + 1e-30) > tol:
                    T_est = T_cal
                    P_est = self.K_prime_val * (T_est**2.5)
                    continue 
                
                converged = True
                P_cal = self.K_prime_val * (T_cal**2.5)
            
            # Luminosity
            rho_cal = self.calculate_density(P_cal, T_cal)
            eps_cal = self.generate_energy(T_cal, rho_cal)
            dL_next = 0.01256637 * (r_next**2) * rho_cal * eps_cal
            d1_next_L = h_step * (dL_next - f_i[3])
            L_cal = v_i[3] + h_step*dL_next - 0.5*d1_next_L
            
            # Store gradients
            dP_next = -self.C_p * (P_cal / T_cal) * (M_cal / (r_next**2))
            grads.append(np.array([dP_next, dT_next, dM_next, dL_next]))
            
            # Print & Store
            energy_str = self._get_energy_label(T_cal, rho_cal)
            print(f"{energy_str:<4} {'CONVEC':<8} {i+1:<4} {r_next:.5f}  {P_cal:.7f}  {T_cal:.7f}  {L_cal:.6f}  {M_cal:.6f}")
            
            new_layer = {
                'i': i+1, 'r': r_next, 'P': P_cal, 'T': T_cal, 'M': M_cal, 'L': L_cal
            }
            layers_out.append(new_layer)
            
            # Check Stop Condition (Matching Radius)
            if r_next >= r_target:
                print(f"--> Matching radius r={r_target:.5f} reached at r={r_next:.5f}")
                reached_target = True
                
            i += 1
            
        return layers_out




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. FINAL STELLAR MODEL EXECUTION (OPTIMIZED PARAMETERS)
    
    # Optimized parameters
    R_opt = 11.0
    L_opt = 75.0
    Tc_opt = 1.95
    
    # Instantiate the model ONCE
    star = StellarModel(M_tot=5.0, R_tot=R_opt, L_tot=L_opt, T_c=Tc_opt, X=0.75, Y=0.22)
    
    print(f"Calculating Final Model: R={star.R_tot}, L={star.L_tot}, Tc={star.T_c_ini}")

    # A. INTEGRATION FROM SURFACE (ENVELOPE) 
    layers_env, h = star.solve_starting_layers()      # START (0-2)
    layers_env = star.solve_phase_A11(layers_env, h)  # A.1.1 (3-5)
    layers_env = star.solve_phase_A12(layers_env, h)  # A.1.2 (M var, L const)
    layers_env = star.solve_phase_A13(layers_env, h)  # A.1.3 (M var, L var)
    
    # Check if convection reached
    if not hasattr(star, 'K_prime_val'):
        print("Error: Convection zone not detected in phase A.1.3.")
        exit()

    # Save Matching Point
    last_rad_layer = layers_env[-1]
    r_match = last_rad_layer['r']
    print(f"\n[INFO] Matching Point at r = {r_match:.5f}")

    # B. INTEGRATION FROM CENTER (CORE) 
    # Use Phase B (outwards)
    h_pos = -h 
    print(f"--> Integrating Core from Center (Phase B)...")
    
    layers_core = star.solve_starting_center_layers(T_c_guess=star.T_c_ini, h_step=h_pos)
    layers_core = star.solve_core_outwards(layers_core, h_step=h_pos, r_target=r_match)

    # 2. UNIFIED FINAL TABLE (EXACT FORMAT)
    print("\n" + "="*110)
    print(f"{'COMPLETE STELLAR MODEL (Surface -> Center)':^110}")
    print("="*110)
    # Header identical to individual phases
    print(f"{'E':<4} {'Phase':<8} {'i':<4} {'r':<10} {'P':<12} {'T':<12} {'L':<10} {'M':<10} {'n+1'}")
    print("-" * 110)

    # PART 1: ENVELOPE (Already ordered Surface -> Match) 
    for layer in layers_env:
        # Retrieve data
        idx = layer['i']
        r = layer['r']
        P = layer['P']
        T = layer['T']
        M = layer['M']
        L = layer['L']
        n_plus_1 = layer.get('n+1', 99.9) # 99.9 if not exists (outer layers)
        
        # Calculate auxiliaries for labels
        rho = star.calculate_density(P, T)
        energy_str = star._get_energy_label(T, rho)
        
        # Phase Label Logic (Reconstruction)
        if idx <= 2:
            phase_str = "START"
        elif 3 <= idx <= 5:
            phase_str = "A.1.1"
        elif abs(L - star.L_tot) < 1e-4: # If L is almost constant
            phase_str = "A.1.2"
        else:
            phase_str = "A.1.3"
        
        # Print format
        n_str = f"{n_plus_1:.3f}" if n_plus_1 < 90 else "-"
        
        print(f"{energy_str:<4} {phase_str:<8} {idx:<4} {r:.5f}  {P:.7f}  {T:.7f}  {L:.6f}  {M:.6f}  {n_str}")

    # Visual separator for Matching Point
    print(f"{'-'*110}")
    print(f"{'--- MATCHING POINT JUMP ---':^110}")
    print(f"{'-'*110}")

    # PART 2: CORE (Ordered Center -> Match, must REVERSE) 
    layers_core_reversed = layers_core[::-1]
    
    # Continue index from envelope
    start_idx = layers_env[-1]['i'] + 1
    
    for j, layer in enumerate(layers_core_reversed):
        idx = start_idx + j
        r = layer['r']
        P = layer['P']
        T = layer['T']
        M = layer['M']
        L = layer['L']
        
        # Calculate real gradients to check n+1 (radiative)
        # FIX: Check to avoid Division by Zero at r=0
        if r > 1e-5:
            dP, dT, _, _ = star.radiative_gradients(r, P, T, M, L) 
            if abs(dT) > 1e-20:
                 n_rad_local = (T/P)*(dP/dT) # This is the radiative n
            else:
                 n_rad_local = 0.0
        else:
            n_rad_local = 0.0 # Center
        
        # Label
        rho = star.calculate_density(P, T)
        energy_str = star._get_energy_label(T, rho)
        
        if r < 1e-3:
            phase_str = "CENTER"
        else:
            phase_str = "CONVEC"
            
        # Print
        n_str = f"{n_rad_local:.3f}" if r > 0.01 else "-"
        
        print(f"{energy_str:<4} {phase_str:<8} {idx:<4} {r:.5f}  {P:.7f}  {T:.7f}  {L:.6f}  {M:.6f}  {n_str}")

    print("="*110)

    # 3. TOTAL ERROR CALCULATION

    final_top = layers_env[-1]
    final_bot = layers_core[-1]
    
    sum_sq_error = 0.0
    print("\nERRORS AT MATCHING POINT:")
    for var in ['P', 'T', 'M', 'L']:
        val_top = final_top[var]
        val_bot = final_bot[var]
        rel_err = abs(val_top - val_bot) / val_top
        sum_sq_error += rel_err**2
        print(f"  Delta {var}: {rel_err*100:.2f}%")
        
    total_error = np.sqrt(sum_sq_error) * 100
    print(f"TOTAL RELATIVE ERROR: {total_error:.4f} %\n")

    # 4. PLOTTING
    
    print("Generating final plot...")
    full_layers = layers_env + layers_core[::-1]
    
    r_arr = np.array([l['r'] for l in full_layers])
    P_arr = np.array([l['P'] for l in full_layers])
    T_arr = np.array([l['T'] for l in full_layers])
    M_arr = np.array([l['M'] for l in full_layers])
    L_arr = np.array([l['L'] for l in full_layers])
    
    plt.figure(figsize=(10, 7))
    plt.plot(r_arr/R_opt, P_arr/P_arr.max(), label=r'Pressure ($P/P_c$)', lw=2)
    plt.plot(r_arr/R_opt, T_arr/T_arr.max(), label=r'Temperature ($T/T_c$)', lw=2, ls='--')
    plt.plot(r_arr/R_opt, M_arr/star.M_tot, label=r'Mass ($M/M_{tot}$)', lw=2, ls='-.')
    plt.plot(r_arr/R_opt, L_arr/L_opt, label=r'Luminosity ($L/L_{tot}$)', lw=2, ls=':', color='red')
    
    plt.axvline(x=r_match/R_opt, color='gray', alpha=0.3)
    plt.text(0.05, 0.5, 'Convective\nCore', transform=plt.gca().transAxes, color='gray')
    plt.text(0.25, 0.5, 'Radiative\nEnvelope', transform=plt.gca().transAxes, color='gray')
    
    plt.xlabel(r'Fractional Radius ($r/R_{tot}$)')
    plt.ylabel('Normalized Amplitude')
    plt.title(f'Final Model (R={R_opt}, L={L_opt}, Tc={Tc_opt})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("final_model_complete.png")
    plt.show()
    print("Plot saved.")



