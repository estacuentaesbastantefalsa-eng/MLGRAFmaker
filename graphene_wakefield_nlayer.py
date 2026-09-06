"""
Estelas de campo (wakefields) excitadas por una carga puntual que se mueve
paralela a N capas de grafeno, con un sustrato aislante opcional.
Implementacion directa del formalismo general de la Sec. 2 de:

Martin-Luna, Bonatto, Bontoiu, Lei, Xia, Resta-Lopez,
"Plasmonic excitations in graphene layers", Chin. J. Phys. 97 (2025) 607-624.
https://doi.org/10.1016/j.cjph.2025.03.030

Para cada (kx,ky) se resuelve el sistema lineal NxN (Eq. 12/19 del paper):

    S_j(k,w) n_j  -  sum_l G_jl(k) n_l  =  B_j(k,w)        j = 1..N

    S_j(k,w) = w(w+i*gamma_j) - alpha_j*k^2 - beta*k^4,   w = kx*v
    G_jl(k)  = 2*pi*n0j*k * [ exp(-k|zj-zl|) - E*exp(-k|zs-zj|)*exp(-k|zs-zl|) ]
    B_j(k,w) = -(2*pi)^2 * n0j*k*Q * [ exp(-k|zj-z0|) - E*exp(-k|zs-zj|)*exp(-k|zs-z0|) ]

    E = (eps_s - 1)/(eps_s + 1)   (E=0 si no hay sustrato)

Igual que en la derivacion de una sola capa, la delta(w - kx*v) que viene de
la carga excitadora se elimina analiticamente mediante la integracion trivial
en w; lo que se resuelve en cada (kx,ky) es la amplitud REDUCIDA n_hat_j(k)
que ya la absorbe (es exactamente el objeto "n_tilde_j(k,z,kx*v)" usado
dentro de las integrales Re[...] de las Eqs. (16)-(17) del paper).

El potencial inducido es  Phi_ind = sum_j Phi_j + Phi_s  (Eq. 3), asi que el
potencial propio del sustrato (Eq. 10) se suma al de las capas (Eq. 9) --
esto importa siempre que se evalue el campo lejos de las propias capas (p.ej.
entre una capa y el sustrato).

Wx = -dPhi/dx y Wz = -dPhi/dz se obtienen derivando bajo el signo integral
(Eq. 18), lo que simplemente multiplica cada termino en espacio-k por
(-i*kx) o por (+-k) con el signo apropiado, antes de hacer la sintesis en
espacio real.
"""

import numpy as np

# ---------------------------------------------------------------- constantes
C_AU = 137.035999084
BOHR_TO_NM = 0.0529177210903
NM_TO_AU = 1.0 / BOHR_TO_NM
FIELD_AU_TO_GVPM = 514.2207

_A0_M = 0.529177210903e-10
N_G_AU = 1.53e20 * _A0_M**2   # densidad superficial de equilibrio del grafeno, u.a.


# ---------------------------------------------------------------------------
# Fisica del modelo (ver Sec. 2 del paper para la derivacion completa):
#
# Cada capa de grafeno j, situada en el plano z=z_j, se modela como un gas
# de electrones 2D (ecuacion de continuidad + Euler linealizadas) el numero de onda transversal relevante es
# simplemente K = sqrt(kx^2+ky^2) (modulo del vector de onda en el plano
# de la capa). El acoplo electrostatico entre capas se hace via el
# propagador de Coulomb 2D en el espacio de Fourier, exp(-K|zj-zl|) (la
# solucion de la ecuacion de Poisson para una lamina de carga plana), en
# vez del propagador cilindrico I_m*K_m usado en el CNT.
#
# El sustrato dielectrico (opcional) se incluye por el metodo de imagenes:
# una capa a distancia |zs-zj| de la interfase se comporta, para efectos
# electrostaticos, como si tuviera una "imagen" de intensidad E=(eps_s-1)/
# (eps_s+1) reflejada al otro lado del sustrato -- de ahi el termino
# -E*exp(-K|zs-zj|)*exp(-K|zs-zl|) restando (o reforzando, segun el signo
# de E) el acoplo directo entre capas.
#
# Aqui siempre se usa friccion finita (gamma>0, tipicamente ~1e-3 u.a.): el
# propio paper de Resta et al indica que ese gamma pequeno solo esta para garantizar la
# convergencia numerica de las integrales de las Eqs. (16)-(17), y que su
# efecto (decaimiento exponencial de la estela lejos del driver) es
# practicamente despreciable para valores tan pequenos.
# ---------------------------------------------------------------------------
class MultilayerGraphene:
    """
    N capas de grafeno + sustrato aislante opcional, excitadas por una
    carga puntual Q que se mueve paralela al eje x en (v t, y0, z0).

    Parametros
    ----------
    z_layers_nm : secuencia de float
        Posicion z de cada capa, en nm (cualquier orden; se ordenan).
    n0_over_ng : float o secuencia de float
        Densidad superficial de cada capa, en unidades de n_g = 1.53e20 m^-2.
        Un solo float se difunde (broadcast) a todas las capas.
    gamma_au : float o secuencia de float
        Parametro de amortiguamiento de cada capa, en u.a. Un solo float
        se difunde a todas las capas.
    v_over_c, Q, z0_nm, y0_nm : parametros del driver.
    eps_s, z_s_nm : permitividad relativa del sustrato y su posicion
        (z <= z_s). Usar eps_s=None (por defecto) si no hay sustrato.
    """

    def __init__(self, z_layers_nm, n0_over_ng=1.0, gamma_au=1e-3,
                 v_over_c=0.05, Q=1.0, z0_nm=0.0, y0_nm=0.0,
                 eps_s=None, z_s_nm=None):
        z_layers_nm = np.atleast_1d(np.asarray(z_layers_nm, dtype=float))
        order = np.argsort(z_layers_nm)
        self.z = z_layers_nm[order] * NM_TO_AU
        self.N = len(self.z)

        # broadcast_to funciona tanto si viene un escalar, una lista de 1
        # elemento o una lista de N elementos (uno por capa); antes, el
        # caso "lista de 1 elemento" con N=1 capa se colaba por la rama
        # float(n0_over_ng), que fallaba porque float() no acepta una lista.
        n0_over_ng = np.broadcast_to(
            np.atleast_1d(np.asarray(n0_over_ng, dtype=float)), (self.N,))[order]
        self.n0 = n0_over_ng * N_G_AU

        gamma_au = np.broadcast_to(
            np.atleast_1d(np.asarray(gamma_au, dtype=float)), (self.N,))[order]
        self.gamma = gamma_au

        # alpha_j: termino de presion de Fermi del fluido de electrones 2D
        # (parte lineal de la relacion de dispersion del plasmon, analoga a
        # una velocidad del sonido al cuadrado); beta*k^4: correccion de
        # presion cuantica (tipo Bohm) que domina a k grande y evita que la
        # dispersion crezca sin limite (misma fisica que en el caso CNT).
        #
        # De donde sale
        #   1) Un gas de electrones fermionico tiene energia cinetica que
        #      depende de la densidad local n, por pura estadistica
        #      cuantica (principio de exclusion de Pauli): a mas densidad,
        #      mas alto hay que llenar los niveles de energia (nivel de
        #      Fermi). En 2D, la energia de Fermi va como eps_F ~ n.
        #   2) En teoria de Thomas-Fermi se trata esa energia cinetica
        #      total, E_kin[n], como un funcional de la densidad local.
        #   3) La derivada funcional dE_kin/dn juega el mismo papel que la
        #      presion en un fluido clasico: si comprimes el gas en un
        #      punto, esa derivada "empuja hacia afuera".
        #   4) Al meter esa derivada en la ecuacion de Euler (Eq. 2 del
        #      paper) y linealizar alrededor de la densidad de equilibrio
        #      n0j (perturbaciones pequenas nj << n0j), esa derivada se
        #      convierte en el termino alpha_j*grad(n), con alpha_j
        #      evaluado en n0j y ya constante.
        #   5) beta*grad[lap(n)] es una correccion mas fina (correccion de
        #      Von Weizsacker en la energia cinetica) que penaliza
        #      gradientes RAPIDOS de densidad 
        self.alpha = np.pi * self.n0          # alpha_j = vFj^2/2 = pi*n0j
        self.beta = 0.25

        self.v = v_over_c * C_AU
        self.Q = Q
        self.z0 = z0_nm * NM_TO_AU
        self.y0 = y0_nm * NM_TO_AU

        if eps_s is not None:
            self.has_substrate = True
            self.E = (eps_s - 1.0) / (eps_s + 1.0)
            self.zs = z_s_nm * NM_TO_AU
        else:
            self.has_substrate = False
            self.E = 0.0
            self.zs = None

    # ------------------------------------------------------ espacio de k
    def _solve_layers(self, KX, KY):
        """Resolucion en bloque del sistema NxN en cada punto de la malla (kx,ky).
        KX, KY: arrays de igual forma (n_kx, n_ky).
        Devuelve n_hat: array complejo (n_kx, n_ky, N).
        """
        K = np.sqrt(KX**2 + KY**2)
        omega = KX * self.v
        shape = K.shape

        M = np.zeros(shape + (self.N, self.N), dtype=complex)
        B = np.zeros(shape + (self.N,), dtype=complex)

        # distancias entre capas |zj-zl|, y al sustrato/driver
        for j in range(self.N):
            # Sj: respuesta inversa de un oscilador armonico amortiguado
            # (dispersion de plasmon de la capa j aislada, con friccion
            # gamma_j tipo Drude) -- identico en estructura al caso CNT,
            # solo que aqui K es el modulo del vector de onda 2D plano en
            # vez del numero de onda "curvo" q_j.
            #
            # gamma_j debe ser >0, aunque sea pequeno para respetar la causalidad:
            #   1) Al resolver la respuesta de un oscilador forzado en
            #      frecuencias, la ecuacion admite tanto soluciones
            #      "avanzadas" (el sistema reacciona ANTES de la fuerza)
            #      como "retardadas" (reacciona DESPUES) -- ambas resuelven
            #      la misma ecuacion diferencial sin friccion.
            #   2) Para forzar la solucion fisica (causal, wake DETRAS del
            #      driver) se usa el truco estandar de dar a omega una
            #      parte imaginaria infinitesimal, omega -> omega + i*eps
            #      (Landau).
            #   3) Una friccion gamma_j>0 pequena hace exactamente ese
            #      papel: desplaza el polo de Sj fuera del eje real omega,
            #      en la direccion que garantiza que la respuesta aparezca
            #      solo para zeta>0 (detras del driver) al hacer la
            #      transformada de Fourier inversa. Por eso el propio paper
            #      usa un gamma finito (aunque pequeno) "para facilitar la
            #      convergencia de las integrales" -- no es solo un truco
            #      numerico, es lo que fija la causalidad de la solucion.
            Sj = omega * (omega + 1j * self.gamma[j]) - self.alpha[j] * K**2 - self.beta * K**4
            dz0 = np.abs(self.z[j] - self.z0)
            sub_term_B = 0.0
            if self.has_substrate:
                # Imagen del driver en el sustrato: el campo que "empuja" a
                # la capa j no es solo el del driver real (exp(-K*dz0)),
                # sino tambien el de su carga imagen al otro lado del
                # sustrato, pesada por el coeficiente de reflexion E.
                sub_term_B = self.E * np.exp(-K * np.abs(self.zs - self.z[j])) * \
                             np.exp(-K * np.abs(self.zs - self.z0))
            # B_j: termino fuente -- cuanto excita el driver (mas su
            # imagen, si hay sustrato) a la densidad de carga de la capa j.
            #
            # la carga del driver se mueve como
            # x=v*t; al hacer la transformada de Fourier temporal de esa
            # fuente aparece un factor delta(omega-kx*v) (Eq. 8 del paper)
            # -- la unica frecuencia que "resuena" con un numero de onda kx
            # es la que corresponde al movimiento uniforme del driver (tipo
            # batido Doppler). Como el sistema Sj*n_hat_j - Gjl*n_hat_l = Bj
            # es LINEAL y Bj ya trae esa delta de fabrica (Eq. 15), la
            # solucion n_hat_j tambien sale multiplicada por la misma
            # delta. Al integrar despues en omega para volver a tiempo
            # real, esa delta hace la integral trivial (sustituye
            # omega->kx*v en todo lo demas y desaparece) -- por eso en el
            # codigo nunca se ve una integral en omega, solo en (kx,ky): lo
            # que se resuelve aqui, B[...,j], es directamente el "peso" que
            # queda tras quitar esa delta 
            B[..., j] = -(2 * np.pi)**2 * self.n0[j] * K * self.Q * \
                        (np.exp(-K * dz0) - sub_term_B)

            for l in range(self.N):
                dzjl = np.abs(self.z[j] - self.z[l])
                sub_term_G = 0.0
                if self.has_substrate:
                    # Igual que arriba pero para el acoplo entre capas j y
                    # l: ademas del Coulomb directo, cada capa "ve" la
                    # imagen de la otra reflejada en el sustrato.
                    sub_term_G = self.E * np.exp(-K * np.abs(self.zs - self.z[j])) * \
                                 np.exp(-K * np.abs(self.zs - self.z[l]))
                # Gjl: acoplo de Coulomb (directo + imagen) entre capas j y
                # l, via el propagador 2D exp(-K|zj-zl|) -- el analogo
                # plano del propagador cilindrico g(aj,al) del CNT.
                Gjl = 2 * np.pi * self.n0[j] * K * (np.exp(-K * dzjl) - sub_term_G)
                if l == j:
                    M[..., j, j] = Sj - Gjl
                else:
                    M[..., j, l] = -Gjl

        # regulariza la esquina (kx,ky)=(0,0) (M es singular ahi; la
        # contribucion al campo de un unico punto es de medida nula en la integral)
        Kflat = K
        singular = Kflat < 1e-12
        if np.any(singular):
            # En K=0 el propagador de Coulomb 2D diverge (la version plana
            # no tiene la regularizacion natural que sí tenia el CNT via
            # I_m*K_m); como es un unico punto de medida nula en la
            # integral 2D en (kx,ky), se sustituye M por la identidad ahi
            # para evitar un NaN numerico sin afectar el resultado fisico.
            M[singular] = np.eye(self.N)[None, :, :]
            B[singular] = 0.0


        # Rsolver M n_hat = B da la respuesta lineal
        # autoconsistente (tipo RPA) de las N capas acopladas ante el
        # driver -- ya incluye el apantallamiento mutuo entre capas y, si
        # existe, el efecto de polarizacion del sustrato dielectrico.
        #
        # Que significa "RPA"?
        #   1) El problema real es que cada electron siente el campo de
        #      TODOS los demas electrones (de su propia capa y de las
        #      otras) -- resolverlo exactamente es inviable.
        #   2) RPA simplifica asumiendo que cada capa responde solo al
        #      campo electrico TOTAL autoconsistente (el externo del driver
        #      mas el que producen las densidades inducidas de todas las
        #      capas), tratado como un campo medio -- sin correlaciones mas
        #      finas entre electrones individuales (de ahi "fase aleatoria":
        #      se asume que esas correlaciones finas se cancelan en
        #      promedio).
        #   3) El sistema Sj*n_hat_j - sum_l Gjl*n_hat_l = Bj (Eq. 12 del
        #      paper) es exactamente esa idea escrita como sistema lineal:
        #      la respuesta de la capa j depende explicitamente de las
        #      densidades de TODAS las demas capas, retroalimentandose
        #      mutuamente hasta llegar a una solucion consistente.
        #   4) Resolver ese sistema con np.linalg.solve ES resolver la
        #      ecuacion autoconsistente de RPA para este problema -- no es
        #      una aproximacion adicional anadida despues.
        n_hat = np.linalg.solve(M, B[..., None])[..., 0]   # (..., N)
        return n_hat, K

    def _sigma_s_hat(self, n_hat, K):
        # No es una capa de electrones fisica, sino la densidad de
        # carga de POLARIZACION inducida en la superficie del dielectrico
        # (metodo de imagenes generalizado): responde tanto al driver como a
        # la carga inducida en cada capa de grafeno, cada una vista a traves
        # de su propio factor de reflexion E y su propia distancia a la
        # interfase.
        """Amplitud reducida de la densidad superficial del sustrato (Eq. 11)."""
        if not self.has_substrate:
            return None
        term_drive = 2 * np.pi * self.Q * np.exp(-K * np.abs(self.zs - self.z0))
        term_layers = np.zeros_like(K, dtype=complex)
        for j in range(self.N):
            term_layers += n_hat[..., j] * np.exp(-K * np.abs(self.zs - self.z[j]))
        return -self.E * (term_drive - term_layers)

    # ------------------------------------------------------- espacio real
    def wakefields_zeta_z(self, zeta_nm, z_nm, kx_max=2.5, ky_max=2.5,
                           n_kx=500, n_ky=250):
        """
        Wx, Wz (GV/m) en el plano (zeta,z) con y=0.
        Devuelve arrays de forma (len(z_nm), len(zeta_nm)).
        """
        zeta = np.asarray(zeta_nm) * NM_TO_AU
        zgrid = np.asarray(z_nm) * NM_TO_AU

        kx = np.linspace(1e-5, kx_max, n_kx)
        ky = np.linspace(1e-5, ky_max, n_ky)
        wkx = _trapz_weights(kx)
        wky = _trapz_weights(ky)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        n_hat, K = self._solve_layers(KX, KY)          # (n_kx,n_ky,N), (n_kx,n_ky)
        sigma_s_hat = self._sigma_s_hat(n_hat, K)        # (n_kx,n_ky) o None

        Cos = np.cos(np.outer(kx, zeta))   # (n_kx, n_zeta)
        Sin = np.sin(np.outer(kx, zeta))

        prefac = 4.0 / (2 * np.pi)**3
        Wx = np.zeros((len(zgrid), len(zeta)))
        Wz = np.zeros((len(zgrid), len(zeta)))

        for iz, z in enumerate(zgrid):
            Dx_total = np.zeros_like(K, dtype=complex)   # coeficiente de (-i*kx)*e^{ikx zeta}
            Dz_total = np.zeros_like(K, dtype=complex)   # coeficiente de e^{ikx zeta} directamente

            for j in range(self.N):
                # decay = exp(-K|z-zj|): el potencial de una lamina de
                # carga plana decae exponencialmente al alejarse de ella en
                # z (a diferencia del CNT, donde la geometria cilindrica da
                # una dependencia I_m/K_m); es literalmente la funcion de
                # Green de Poisson 2D+1D evaluada a un lado u otro del plano.
                #
                #
                #   1) Fuera de las laminas cargadas, el potencial satisface
                #      la ecuacion de Laplace: lap(Phi)=0.
                #   2) Si la dependencia en (x,y) de una componente de
                #      Fourier es e^{i(kx x+ky y)}, entonces
                #      d2/dx2+d2/dy2 -> -K^2, y la ecuacion se reduce a una
                #      EDO en z sola: phi''(z) - K^2*phi(z) = 0.
                #   3) Las soluciones generales son e^{+Kz} y e^{-Kz}. Pero
                #      fisicamente el potencial debe anularse en z->+-inf, asi
                #      que solo sobrevive la solucion que decae a cada
                #      lado: e^{-Kz} para z>0 y e^{+Kz} para z<0, que se
                #      escribe compacto como e^{-K|z|}.
                #   4) La amplitud exacta (el factor 2*pi*n0j que aparece
                #      en Gjl y Bj) se fija integrando Poisson a traves de
                #      la lamina cargada (de z=0- a z=0+), lo que da el
                #      salto caracteristico del campo electrico normal al
                #      cruzar una lamina de carga.
                decay = np.exp(-K * np.abs(z - self.z[j]))
                # sgn = signo de (z - z_j): el campo ELECTRICO (a diferencia
                # del potencial) es discontinuo al cruzar una lamina de
                # carga -- apunta "hacia afuera" en ambos lados, como el
                # campo de un plano infinito cargado -- de ahi que dD/dz
                # cambie de signo segun estemos por encima o por debajo de
                # la capa j.
                sgn = np.sign(z - self.z[j]) or 1.0
                # D_j(k,z) = -(2pi/k) n_hat_j * decay  -> se usa para Wx via -i*kx*D_j
                Dj = -(2 * np.pi / K) * n_hat[..., j] * decay
                Dx_total += Dj
                # -dD_j/dz = -2*pi*sgn*n_hat_j*decay  (k se cancela, ver docstring del modulo)
                Dz_total += -2 * np.pi * sgn * n_hat[..., j] * decay

            if self.has_substrate:
                # Misma logica que arriba pero para el potencial de
                # polarizacion inducido en el sustrato (Eq. 10): tambien
                # decae exponencialmente y produce su propio salto de campo
                # al cruzar z=zs.
                decay_s = np.exp(-K * np.abs(z - self.zs))
                sgn_s = np.sign(z - self.zs) or 1.0
                # D_s(k,z) = +(2*pi/k) * sigma_s_hat * decay_s   (Eq. 10)
                Dx_total += (2 * np.pi / K) * sigma_s_hat * decay_s
                # -dD_s/dz = 2*pi*sgn_s*sigma_s_hat*decay_s   (k se cancela)
                Dz_total += 2 * np.pi * sgn_s * sigma_s_hat * decay_s

            # Wx = -dPhi/dx: en espacio de Fourier, derivar respecto a x
            # equivale a multiplicar por (i*kx); Wz = -dPhi/dz ya se obtuvo
            # derivando "a mano" arriba (el modulo K se cancela con el
            # 1/K de la amplitud del potencial, ver comentarios de Dj).
            Ax = -1j * KX * Dx_total
            Az = Dz_total

            for A, out in ((Ax, Wx), (Az, Wz)):
                A_re = A.real * wky[None, :] * wkx[:, None]
                A_im = A.imag * wky[None, :] * wkx[:, None]
                A_re_kx = A_re.sum(axis=1)
                A_im_kx = A_im.sum(axis=1)
                # Sintesis de Fourier de vuelta al espacio real zeta:
                # exactamente el mismo reparto Re->coseno / Im->seno (via
                # e^{i kx zeta} = cos+i*sin) que separa la parte reactiva
                # (dispersiva, sin perdidas) de la parte disipativa
                # (asociada a la friccion gamma_j) de la respuesta, igual
                # que en el caso del CNT.
                out[iz, :] = prefac * (A_re_kx @ Cos - A_im_kx @ Sin)

        return Wx * FIELD_AU_TO_GVPM, Wz * FIELD_AU_TO_GVPM

    def wakefields_zeta_y(self, zeta_nm, y_nm, z_nm, kx_max=2.5, ky_max=2.5,
                           n_kx=500, n_ky=250):
        """
        Wx, Wy, Wz (GV/m) en el plano (zeta, y), a un z = z_nm FIJO (p.ej.
        la altura de una capa elegida, o el propio z0 del driver). Es el
        complemento de wakefields_zeta_z (que fija y=0 y varia z).

        Devuelve arrays de forma (len(y_nm), len(zeta_nm)).
        """
        zeta = np.asarray(zeta_nm) * NM_TO_AU
        ygrid = np.asarray(y_nm) * NM_TO_AU
        zval = z_nm * NM_TO_AU

        kx = np.linspace(1e-5, kx_max, n_kx)
        ky = np.linspace(1e-5, ky_max, n_ky)
        wkx = _trapz_weights(kx)
        wky = _trapz_weights(ky)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        n_hat, K = self._solve_layers(KX, KY)
        sigma_s_hat = self._sigma_s_hat(n_hat, K)

        # Dpot = la propia amplitud del potencial (sin derivar aun) -- se
        # usa para Wx (via -i*kx) y Wy (via +ky, con seno en vez de coseno).
        # Dz_total = la amplitud de la derivada en z (ya derivada, misma
        # construccion que en wakefields_zeta_z) -- se usa para Wz.
        # Fisica: Wy = -dPhi/dy se obtiene igual que Wx pero derivando
        # respecto a la coordenada transversal y (multiplicar por i*ky) y
        # usando la transformada seno en y en vez de coseno -- consistente
        # con que el campo transversal Wy debe anularse en y=0 quando el
        # driver esta centrado en y0=0 (simetria de reflexion en y),
        # exactamente como corresponde a una funcion impar representada
        # por senos.
        Dpot = np.zeros_like(K, dtype=complex)
        Dz_total = np.zeros_like(K, dtype=complex)
        for j in range(self.N):
            decay = np.exp(-K * np.abs(zval - self.z[j]))
            sgn = np.sign(zval - self.z[j])
            sgn = sgn if sgn != 0 else 1.0
            Dpot += -(2 * np.pi / K) * n_hat[..., j] * decay
            Dz_total += -2 * np.pi * sgn * n_hat[..., j] * decay

        if self.has_substrate:
            decay_s = np.exp(-K * np.abs(zval - self.zs))
            sgn_s = np.sign(zval - self.zs)
            sgn_s = sgn_s if sgn_s != 0 else 1.0
            Dpot += (2 * np.pi / K) * sigma_s_hat * decay_s
            Dz_total += 2 * np.pi * sgn_s * sigma_s_hat * decay_s

        Cos_x = np.cos(np.outer(kx, zeta))   # (n_kx, n_zeta)
        Sin_x = np.sin(np.outer(kx, zeta))
        Cos_y = np.cos(np.outer(ky, ygrid))  # (n_ky, n_y)
        Sin_y = np.sin(np.outer(ky, ygrid))
        prefac = 4.0 / (2 * np.pi)**3

        def transform(A, y_trig):
            # A: amplitud compleja (n_kx,n_ky) en espacio-k; y_trig: Cos_y o Sin_y
            inner = (A * wky[None, :]) @ y_trig        # (n_kx, n_y) complejo
            re = inner.real * wkx[:, None]
            im = inner.imag * wkx[:, None]
            return prefac * (re.T @ Cos_x - im.T @ Sin_x)   # (n_y, n_zeta)

        Ax_coef = -1j * KX * Dpot
        Ay_coef = KY * Dpot
        Az_coef = Dz_total

        Wx = transform(Ax_coef, Cos_y)
        Wy = transform(Ay_coef, Sin_y)
        Wz = transform(Az_coef, Cos_y)

        return (Wx * FIELD_AU_TO_GVPM, Wy * FIELD_AU_TO_GVPM, Wz * FIELD_AU_TO_GVPM)

    def wx_along_axis(self, zeta_nm, **kwargs):
        """Wx(zeta) evaluado en la propia posicion transversal del driver (y0,z0)."""
        return self.wakefields_zeta_z(zeta_nm, [self.z0 / NM_TO_AU], **kwargs)[0][0]

    def wx_max_vs_velocity(self, v_over_c_array, zeta_nm, **kwargs):
        """Barre la velocidad del driver, devuelve max|Wx| a lo largo del eje
        del driver para cada v (curva tipo Fig. 8/16/19). Reconstruye la
        velocidad del driver en cada llamada."""
        out = np.zeros(len(v_over_c_array))
        for i, vc in enumerate(v_over_c_array):
            self.v = vc * C_AU
            wx = self.wx_along_axis(zeta_nm, **kwargs)
            out[i] = np.max(wx)
        return out


def _trapz_weights(x):
    w = np.zeros_like(x)
    d = np.diff(x)
    w[0] = d[0] / 2
    w[-1] = d[-1] / 2
    w[1:-1] = (d[:-1] + d[1:]) / 2
    return w
