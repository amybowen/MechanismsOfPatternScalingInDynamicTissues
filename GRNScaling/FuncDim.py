"""

Functions to calculate simulation parameters
check the parameters are biologically realistic
and to convert between dimensions

"""

def calculate_useful_simulation_parameters(N,L0,dt_factor,D,k,g_max,k1,k2,sourceSize,t_max,store_no):
    # calculate values
    N_1 = N-1
    N_2 = N-2
    dy = L0/(N-1) # space step
    w0 = round(sourceSize/dy)+1 # the index for the edge of the source
    dt = dt_factor*(2/(4*D/(dy**2) + max(g_max,k,k1,k2)))   # setting dt according to k and g_max, via the vN stability criterion
    count_max = round(t_max/dt) # the number of time points that will be computed - round this up
    dt_store = dt * store_no
    return N_1, N_2, dy, w0, dt, count_max, dt_store



def calculate_useful_simulation_parameters_no_growth(N,L0,dt_factor,D,k,k1,k2,sourceSize,store_no_ng,t_max_factor = 10):
    N_1 = N-1
    N_2 = N-2
    dy = L0/(N-1) # space step
    w0 = round(sourceSize/dy)+1 # the index for the edge of the source
    dt_ng = dt_factor*(2/(4*D/(dy**2) + max(k,k1,k2)))   # setting dt according to k and g_max, via the vN stability criterion
    t_max_ng = t_max_factor*(1/k1 + 1/k2)
    count_max_ng = round(t_max_ng/dt_ng) # the number of time points that will be computed - round this up
    dt_store_ng = dt_ng * store_no_ng
    return N_1, N_2, dy, w0, dt_ng, count_max_ng, dt_store_ng

def von_Neumann_and_growth_check(dt, D, dy, k, g_max, k1, k2, epsilon):
    """ A function to perform the von Neumann stability and growth stability checks to ensure numerical stability for the concentrations and growth. """
    # %%%%%%%%%%%% STABILITY CHECK - von Neumann and growth %%%%%%%%%%%%
    stability = dt * (4*D/((dy)**2) + k+g_max)
    # print('diffusion part of error',4*D/((dy)**2))
    # print('k+g_max part of error',k+g_max)
    gStability = g_max*dt/(1+epsilon)
    print("growth stability",gStability)
    print("vN",stability)
    n1stability = k1*dt
    n2stability = k2*dt
    if stability < 2 and gStability <1 and n1stability < 1 and n2stability < 1:
        print('Stable')
    else:
        print('Unstable')
    return

def growth_dy_check(L0,Lf,N):
    """ A function to check that the space steps are small enough. """
    dy = L0/(N-1) # space step
    dy_f = Lf/(N-1)
    print("initial dy, final dy", dy, dy_f)
    return


# def check_biologically_realistic(Literature_values, k, lam, g_max, L0, epsilon, V1, V2, k1, k2):
#     print('k', 'real', Literature_values['morphogen']['degradation_rate']['value'], 'sim', k)
#     print('lambda',  'real', Literature_values['morphogen']['decay_length']['value'], 'sim', lam)
#     print('growth rate', 'real', Literature_values['tissue']['growth_rate']['value'], 'sim', g_max)
#     print('tissue length', 'real', Literature_values['tissue']['tissue_length']['value'], 'sim', L0)
#     print('anisotropy parameter', 'real', Literature_values['tissue']['anisotropy_parameter']['value'], 'sim', epsilon)
#     print('GRN production rates', 'real', Literature_values['GRN']['production_rate']['value'], 'sim', V1, V2)
#     print('GRN degradation rates', 'real', Literature_values['GRN']['degradation_rate']['value'], 'sim', k1, k2)
#     return


def check_biologically_realistic(literature_values, k, lam, g_max, L0, epsilon, V1, V2, k1, k2):
    parameter_map = [
        ("morphogen", "degradation_rate", "k", k),
        ("morphogen", "decay_length", "lambda", lam),
        ("tissue", "growth_rate", "growth rate", g_max),
        ("tissue", "tissue_length", "tissue length", L0),
        ("tissue", "anisotropy_parameter", "anisotropy parameter", epsilon),
        ("GRN", "production_rate", "GRN production rates", (V1, V2)),
        ("GRN", "degradation_rate", "GRN degradation rates", (k1, k2)),
    ]

    for category, parameter, label, sim_value in parameter_map:
        entry = literature_values[category][parameter]

        print(label, "sim", sim_value)

        if isinstance(entry, list):
            for item in entry:
                value = item.get("value", "missing")
                units = item.get("units", "")
                source = item.get("source", "unknown source")
                system = item.get("system", "unknown system")

                print("  real", value, units, "system", system, "source", source)

        else:
            value = entry.get("value", "missing")
            units = entry.get("units", "")
            source = entry.get("source", "unknown source")

            print("  real", value, units, "source", source)


def find_nd4_parameters(V1,k1,N1star,Mstar,V2,k2,N2star,VM,k,D,g):
    Dtilde = D / k1
    alpha1 = V1 /( k1* N1star)
    alpha2 = V2 /( k1* N2star) 
    nu2 = k2 / k1
    ktilde = k / k1
    vtilde = VM / (k1*Mstar)
    gtilde = g / k1
    return Dtilde, alpha1, alpha2, nu2, vtilde, ktilde, gtilde

def find_nd4_parameters_from_nd3(v1,k1,v2,k2,vm,k,D,g):
    Dtilde = D / k1
    alpha1 = v1 /k1
    alpha2 = v2 /k1
    nu2 = k2 / k1
    ktilde = k / k1
    vtilde = vm /k1
    gtilde = g /k1
    return Dtilde, alpha1, alpha2, nu2, vtilde, ktilde, gtilde


def find_nd3_parameters(VM, Mstar, V1, N1star, V2, N2star):
    vm = VM/Mstar
    v1 = V1/N1star
    v2 = V2/N2star
    return vm, v1, v2


def find_dim_parameters_from_4nd(Dtilde, alpha1, alpha2, nu2, vtilde, ktilde, gtilde, k1, N1star, Mstar, N2star):
    D = Dtilde * k1
    V1 = alpha1 * k1 * N1star
    V2 = alpha2 * k1 * N2star
    k = ktilde * k1
    VM = vtilde * k1 * Mstar
    k2 = nu2 * k1
    g = gtilde * k1
    return D, V1, V2, k, VM, g, k2