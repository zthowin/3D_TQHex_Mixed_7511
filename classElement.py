#----------------------------------------------------------------------------------------
# Module housing top-level element class.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 13, 2022
#----------------------------------------------------------------------------------------
import sys

try:
  import numpy as np
except ImportError:
  sys.exit("MODULE WARNING. NumPy not installed.")

try:
  import Lib
except ImportError:
  sys.exit("MODULE WARNING. 'Lib.py' not found, check configuration.")

try:
  import _ElementVariables
except ImportError:
  sys.exit("MODULE WARNING. '_ElementVariables.py' not found, check configuration.")

try:
  import _ElementForces
except ImportError:
  sys.exit("MODULE WARNING. '_ElementForces.py' not found, check configuration.")

try:
  import _ElementTangents
except ImportError:
  sys.exit("MODULE WARNING. '_ElementTangents.py' not found, check configuration.")

@Lib.add_methods_from(_ElementVariables, _ElementForces, _ElementTangents)

class Element:
    
    def __init__(self, a_GaussOrder=None, a_ID=None):
        # Initialize Gauss quadrature order.
        self.set_Gauss_Order(a_GaussOrder)
        # Initialize element ID.
        self.set_Element_ID(a_ID)
        return
    
    def set_Gauss_Order(self, a_Order):
        # Set the gauss quadrature order.
        self.Gauss_Order = a_Order
        return

    def set_Element_ID(self, a_ID):
        # Set the element number.
        self.ID = a_ID
        return

    def set_Gauss_Points(self, Parameters):
        # Initialize the Gauss quadrature points.
        # Assumes Parameters.numDim = 3
        if self.Gauss_Order == 2:
            const        = 1/np.sqrt(3)
            self.points  = np.array([[-const, -const, -const],
                                     [const,  -const, -const],
                                     [const,   const, -const],
                                     [-const,  const, -const],
                                     [-const, -const,  const],
                                     [const,  -const,  const],
                                     [const,   const,  const],
                                     [-const,  const,  const]],
                                    dtype=Parameters.float_dtype)
        elif self.Gauss_Order == 3:
            const        = np.sqrt(3/5)
            self.points  = np.array([[-const, -const, -const],
                                     [const,  -const, -const],
                                     [const,   const, -const],
                                     [-const,  const, -const],
                                     [-const, -const,  const],
                                     [const,  -const,  const],
                                     [const,   const,  const],
                                     [-const,  const,  const],
                                     [0,      -const, -const],
                                     [const,       0, -const],
                                     [0,       const, -const],
                                     [-const,      0, -const],
                                     [0,      -const,  const],
                                     [const,       0,  const],
                                     [0,       const,  const],
                                     [-const,      0,  const],
                                     [-const, -const,      0],
                                     [const,  -const,      0],
                                     [const,   const,      0],
                                     [0,           0, -const],
                                     [0,           0,  const],
                                     [0,      -const,      0],
                                     [0,       const,      0],
                                     [-const,      0,      0],
                                     [const,       0,      0],
                                     [0,           0,      0]],
                                    dtype=Parameters.float_dtype)
        else:
            sys.exit("ERROR. Only trilinear and triquadratic elements have been implemented; check quadrature order.")
        return

    def set_Gauss_Weights(self, Parameters):
        # Initialize the Gauss quadrature weights.
        if self.Gauss_Order == 2:
            self.weights = np.ones(Parameters.numGauss, dtype=Parameters.float_dtype)
        elif self.Gauss_Order == 3:
            const1           = 5/9
            const2           = 8/9
            self.weights     = np.zeros(Parameters.numGauss, dtype=Parameters.float_dtype)
            self.weights[0]  = b**3
            self.weights[1]  = b**3
            self.weights[2]  = b**3
            self.weights[3]  = b**3
            self.weights[4]  = b**3
            self.weights[5]  = b**3
            self.weights[6]  = b**3
            self.weights[7]  = b**3
            self.weights[8]  = c*b**2
            self.weights[9]  = c*b**2
            self.weights[10] = c*b**2
            self.weights[11] = c*b**2
            self.weights[12] = c*b**2
            self.weights[13] = c*b**2
            self.weights[14] = c*b**2
            self.weights[15] = c*b**2
            self.weights[16] = c*b**2
            self.weights[17] = c*b**2
            self.weights[18] = c*b**2
            self.weights[19] = c*b**2
            self.weights[20] = b*c**2
            self.weights[21] = b*c**2
            self.weights[22] = b*c**2
            self.weights[23] = b*c**2
            self.weights[24] = b*c**2
            self.weights[25] = b*c**2
            self.weights[26] = c**3
        else:
            sys.exit("ERROR. Only trilinear and triquadratic elements have been implemented; check quadrature order.")
        return

    def set_Coordinates(self, a_Coords):
        # Initialize element coordinates.
        self.coordinates = a_Coords
        return

    def evaluate_Shape_Functions(self, Parameters):
        # Initialize the shape functions used for interpolation.        
        #--------------------------------
        # Grab local element coordinates.
        #--------------------------------
        self.xi   = self.points[:,0]
        self.eta  = self.points[:,1]
        self.zeta = self.points[:,2]
        #---------
        # Set N_a.
        #---------
        self.N1L = (1 - self.xi)*(1 - self.eta)*(1 - self.zeta)/8
        self.N2L = (1 + self.xi)*(1 - self.eta)*(1 - self.zeta)/8
        self.N3L = (1 + self.xi)*(1 + self.eta)*(1 - self.zeta)/8
        self.N4L = (1 - self.xi)*(1 + self.eta)*(1 - self.zeta)/8
        self.N5L = (1 - self.xi)*(1 - self.eta)*(1 + self.zeta)/8
        self.N6L = (1 + self.xi)*(1 - self.eta)*(1 + self.zeta)/8
        self.N7L = (1 + self.xi)*(1 + self.eta)*(1 + self.zeta)/8
        self.N8L = (1 - self.xi)*(1 + self.eta)*(1 + self.zeta)/8
        #-----------------------------
        # Build shape function matrix.
        #-----------------------------
        self.NuL = np.zeros((Parameters.numGauss, Parameters.numDim, Parameters.numElDOF), dtype=Parameters.float_dtype)
        for i in range(3):
            self.NuL[:, i, 0 + i]  = self.N1L
            self.NuL[:, i, 3 + i]  = self.N2L
            self.NuL[:, i, 6 + i]  = self.N3L
            self.NuL[:, i, 9 + i]  = self.N4L
            self.NuL[:, i, 12 + i] = self.N5L
            self.NuL[:, i, 15 + i] = self.N6L
            self.NuL[:, i, 18 + i] = self.N7L
            self.NuL[:, i, 21 + i] = self.N8L
        #----------------------------------------------------
        # Build shape function matrix for pressure if needed.
        #----------------------------------------------------
        if Parameters.numElDOFP > 0:
            if Parameters.numElDOFP == 1:
                self.Np = np.ones((Parameters.numGauss), dtype=Parameters.float_dtype)
            elif Parameters.numElDOFP == 4:
                self.Np      = np.ones((Parameters.numGauss, Parameters.numElDOFP), dtype=Parameters.float_dtype)
                self.Np[:,1] = self.xi[:]
                self.Np[:,2] = self.eta[:]
                self.Np[:,3] = self.zeta[:]
            elif Parameters.numElDOFP == 8:
                self.Np = np.copy(self.NuL)
            else:
                sys.exit("ERROR. Number of pressure degrees of freedom must be 1, 4, or 8. Check parameters.")
        #--------------------------------------
        # Calculate shape function derivatives.
        #--------------------------------------
        if Parameters.numGauss == 8:
            self.getLinearDerivatives(Parameters)
        elif Parameters.numGauss == 27:
            self.getQuadraticDerivatives(Parameters)
        #-------------------
        # Compute jacobians.
        #-------------------
        self.get_Jacobian(Parameters)
        #----------------------------------
        # Compute shape function gradients.
        #----------------------------------
        self.dN1_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN1_dxi, self.dN1_deta, self.dN1_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN2_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN2_dxi, self.dN2_deta, self.dN2_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN3_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN3_dxi, self.dN3_deta, self.dN3_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN4_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN4_dxi, self.dN4_deta, self.dN4_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN5_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN5_dxi, self.dN5_deta, self.dN5_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN6_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN6_dxi, self.dN6_deta, self.dN6_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN7_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN7_dxi, self.dN7_deta, self.dN7_dzeta]).T, dtype=Parameters.float_dtype)
        self.dN8_dx = np.einsum('...bI, ...b -> ...I', self.Jeinv, np.array([self.dN8_dxi, self.dN8_deta, self.dN8_dzeta]).T, dtype=Parameters.float_dtype)
        #--------------------------------------
        # Construct strain-displacement matrix.
        #--------------------------------------
        self.Bu = np.zeros((Parameters.numGauss, Parameters.numDim**2, Parameters.numElDOF), dtype=Parameters.float_dtype)

        for i in range(3):
            self.Bu[:, i, 0]  = self.dN1_dx[:,i]
            self.Bu[:, i, 3]  = self.dN2_dx[:,i]
            self.Bu[:, i, 6]  = self.dN3_dx[:,i]
            self.Bu[:, i, 9]  = self.dN4_dx[:,i]
            self.Bu[:, i, 12] = self.dN5_dx[:,i]
            self.Bu[:, i, 15] = self.dN6_dx[:,i]
            self.Bu[:, i, 18] = self.dN7_dx[:,i]
            self.Bu[:, i, 21] = self.dN8_dx[:,i]

        for i in range(3,6):
            self.Bu[:, i, 1]  = self.dN1_dx[:,i-3]
            self.Bu[:, i, 4]  = self.dN2_dx[:,i-3]
            self.Bu[:, i, 7]  = self.dN3_dx[:,i-3]
            self.Bu[:, i, 10] = self.dN4_dx[:,i-3]
            self.Bu[:, i, 13] = self.dN5_dx[:,i-3]
            self.Bu[:, i, 16] = self.dN6_dx[:,i-3]
            self.Bu[:, i, 19] = self.dN7_dx[:,i-3]
            self.Bu[:, i, 22] = self.dN8_dx[:,i-3]

        for i in range(6,9):
            self.Bu[:, i, 2]  = self.dN1_dx[:,i-6]
            self.Bu[:, i, 5]  = self.dN2_dx[:,i-6]
            self.Bu[:, i, 8]  = self.dN3_dx[:,i-6]
            self.Bu[:, i, 11] = self.dN4_dx[:,i-6]
            self.Bu[:, i, 14] = self.dN5_dx[:,i-6]
            self.Bu[:, i, 17] = self.dN6_dx[:,i-6]
            self.Bu[:, i, 20] = self.dN7_dx[:,i-6]
            self.Bu[:, i, 23] = self.dN8_dx[:,i-6]
        
        return

    def evaluate_Shape_Functions_2D(self, Parameters):
        # Create a 2D planar element for traction boundary condition.
        #---------
        # Set N_a.
        #---------
        self.N5_2D = (1 - self.xi[4:8])*(1 - self.eta[4:8])/4
        self.N6_2D = (1 + self.xi[4:8])*(1 - self.eta[4:8])/4
        self.N7_2D = (1 + self.xi[4:8])*(1 + self.eta[4:8])/4
        self.N8_2D = (1 - self.xi[4:8])*(1 + self.eta[4:8])/4
        #-----------------------------
        # Build shape function matrix.
        #-----------------------------
        self.Nu_2D = np.zeros((4, Parameters.numDim, Parameters.numElDOF), dtype=Parameters.float_dtype)
        for i in range(3):
            self.Nu_2D[:, i, 12 + i] = self.N5_2D
            self.Nu_2D[:, i, 15 + i] = self.N6_2D
            self.Nu_2D[:, i, 18 + i] = self.N7_2D
            self.Nu_2D[:, i, 21 + i] = self.N8_2D
        #----------------------------------
        # Calculate derivatives w.r.t. \xi.
        #----------------------------------
        self.dN5_dxi_2D = -(1/4)*(1 - self.eta[4:8])
        self.dN6_dxi_2D = -self.dN5_dxi_2D
        self.dN7_dxi_2D = (1/4)*(1 + self.eta[4:8])
        self.dN8_dxi_2D = -self.dN7_dxi_2D
        
        self.dN_dxi_2D      = np.zeros((4,4), dtype=Parameters.float_dtype)
        self.dN_dxi_2D[:,0] = self.dN5_dxi_2D
        self.dN_dxi_2D[:,1] = self.dN6_dxi_2D
        self.dN_dxi_2D[:,2] = self.dN7_dxi_2D
        self.dN_dxi_2D[:,3] = self.dN8_dxi_2D
        #-----------------------------------
        # Calculate derivatives w.r.t. \eta.
        #-----------------------------------
        self.dN5_deta_2D = -(1/4)*(1 - self.xi[4:8])
        self.dN6_deta_2D = -(1/4)*(1 + self.xi[4:8])
        self.dN7_deta_2D = -self.dN6_deta_2D
        self.dN8_deta_2D = -self.dN5_deta_2D
        
        self.dN_deta_2D      = np.zeros((4,4), dtype=Parameters.float_dtype)
        self.dN_deta_2D[:,0] = self.dN5_deta_2D
        self.dN_deta_2D[:,1] = self.dN6_deta_2D
        self.dN_deta_2D[:,2] = self.dN7_deta_2D
        self.dN_deta_2D[:,3] = self.dN8_deta_2D
        #------------------------
        # Calculate the jacobian.
        #------------------------
        self.get_Jacobian_2D(Parameters)

        return

    def getLinearDerivatives(self, Parameters):
        # Compute the derivatives of the linear shape functions.
        #----------------------------------
        # Calculate derivatives w.r.t. \xi.
        #----------------------------------
        self.dN1_dxi = -(1/8)*(1 - self.eta)*(1 - self.zeta)
        self.dN2_dxi = -self.dN1_dxi
        self.dN3_dxi = (1/8)*(1 + self.eta)*(1 - self.zeta)
        self.dN4_dxi = -self.dN3_dxi
        self.dN5_dxi = -(1/8)*(1 - self.eta)*(1 + self.zeta)
        self.dN6_dxi = -self.dN5_dxi
        self.dN7_dxi = (1/8)*(1 + self.eta)*(1 + self.zeta)
        self.dN8_dxi = -self.dN7_dxi
        
        self.dN_dxi      = np.zeros((Parameters.numGauss,Parameters.numGauss), dtype=Parameters.float_dtype)
        self.dN_dxi[:,0] = self.dN1_dxi
        self.dN_dxi[:,1] = self.dN2_dxi
        self.dN_dxi[:,2] = self.dN3_dxi
        self.dN_dxi[:,3] = self.dN4_dxi
        self.dN_dxi[:,4] = self.dN5_dxi
        self.dN_dxi[:,5] = self.dN6_dxi
        self.dN_dxi[:,6] = self.dN7_dxi
        self.dN_dxi[:,7] = self.dN8_dxi
        #-----------------------------------
        # Calculate derivatives w.r.t. \eta.
        #-----------------------------------
        self.dN1_deta = -(1/8)*(1 - self.xi)*(1 - self.zeta)
        self.dN2_deta = -(1/8)*(1 + self.xi)*(1 - self.zeta)
        self.dN3_deta = -self.dN2_deta
        self.dN4_deta = -self.dN1_deta
        self.dN5_deta = -(1/8)*(1 - self.xi)*(1 + self.zeta)
        self.dN6_deta = -(1/8)*(1 + self.xi)*(1 + self.zeta)
        self.dN7_deta = -self.dN6_deta
        self.dN8_deta = -self.dN5_deta
        
        self.dN_deta      = np.zeros((Parameters.numGauss,Parameters.numGauss), dtype=Parameters.float_dtype)
        self.dN_deta[:,0] = self.dN1_deta
        self.dN_deta[:,1] = self.dN2_deta
        self.dN_deta[:,2] = self.dN3_deta
        self.dN_deta[:,3] = self.dN4_deta
        self.dN_deta[:,4] = self.dN5_deta
        self.dN_deta[:,5] = self.dN6_deta
        self.dN_deta[:,6] = self.dN7_deta
        self.dN_deta[:,7] = self.dN8_deta
        #------------------------------------
        # Calculate derivatives w.r.t. \zeta.
        #------------------------------------
        self.dN1_dzeta = -(1/8)*(1 - self.xi)*(1 - self.eta)
        self.dN2_dzeta = -(1/8)*(1 + self.xi)*(1 - self.eta)
        self.dN3_dzeta = -(1/8)*(1 + self.xi)*(1 + self.eta)
        self.dN4_dzeta = -(1/8)*(1 - self.xi)*(1 + self.eta)
        self.dN5_dzeta = -self.dN1_dzeta
        self.dN6_dzeta = -self.dN2_dzeta
        self.dN7_dzeta = -self.dN3_dzeta
        self.dN8_dzeta = -self.dN4_dzeta
        
        self.dN_dzeta      = np.zeros((Parameters.numGauss,Parameters.numGauss), dtype=Parameters.float_dtype)
        self.dN_dzeta[:,0] = self.dN1_dzeta
        self.dN_dzeta[:,1] = self.dN2_dzeta
        self.dN_dzeta[:,2] = self.dN3_dzeta
        self.dN_dzeta[:,3] = self.dN4_dzeta
        self.dN_dzeta[:,4] = self.dN5_dzeta
        self.dN_dzeta[:,5] = self.dN6_dzeta
        self.dN_dzeta[:,6] = self.dN7_dzeta
        self.dN_dzeta[:,7] = self.dN8_dzeta

     def getQuadraticDerivatives(self, Parameters):
        # Compute the derivatives of the quadratic shape functions.
        #----------------------------------
        # Calculate derivatives w.r.t. \xi.
        #----------------------------------
        self.dN_dxi       = np.zeros((Parameters.numGauss, Parameters.numGauss), dtype=Parameters.float_dtype)
        #
        self.dN_dxi[:,0]  = (1/8)*self.eta*self.zeta*(self.eta - 1)*(self.zeta - 1)*(2*self.xi - 1)
        self.dN_dxi[:,1]  = (1/8)*self.eta*self.zeta*(self.eta - 1)*(self.zeta - 1)*(2*self.xi + 1)
        self.dN_dxi[:,2]  = (1/8)*self.eta*self.zeta*(self.eta + 1)*(self.zeta - 1)*(2*self.xi + 1)
        self.dN_dxi[:,3]  = (1/8)*self.eta*self.zeta*(self.eta + 1)*(self.zeta - 1)*(2*self.xi - 1)
        self.dN_dxi[:,4]  = (1/8)*self.eta*self.zeta*(self.eta - 1)*(self.zeta + 1)*(2*self.xi - 1)
        self.dN_dxi[:,5]  = (1/8)*self.eta*self.zeta*(self.eta - 1)*(self.zeta + 1)*(2*self.xi + 1)
        self.dN_dxi[:,6]  = (1/8)*self.eta*self.zeta*(self.eta + 1)*(self.zeta + 1)*(2*self.xi + 1)
        self.dN_dxi[:,7]  = (1/8)*self.eta*self.zeta*(self.eta + 1)*(self.zeta + 1)*(2*self.xi - 1)
        #
        self.dN_dxi[:,8]  = (1/4)*self.eta*self.zeta*(self.eta - 1)*(self.zeta - 1)*(-2*self.xi)
        self.dN_dxi[:,9]  = (1/4)*(1 - self.eta**2)*self.zeta*(self.zeta - 1)*(2*self.xi + 1)
        self.dN_dxi[:,10] = (1/4)*self.eta*self.zeta*(self.eta + 1)*(self.zeta - 1)*(-2*self.xi)
        self.dN_dxi[:,11] = (1/4)*(1 - self.eta**2)*self.zeta*(self.zeta - 1)*(2*self.xi - 1)
        self.dN_dxi[:,12] = (1/4)*self.eta*self.zeta*(self.eta - 1)*(self.zeta + 1)*(-2*self.xi)
        self.dN_dxi[:,13] = (1/4)*(1 - self.eta**2)*self.zeta*(self.zeta + 1)*(2*self.xi + 1)
        self.dN_dxi[:,14] = (1/4)*self.eta*self.zeta*(self.eta + 1)*(self.zeta + 1)*(-2*self.xi)
        self.dN_dxi[:,15] = (1/4)*(1 - self.eta**2)*self.zeta*(self.zeta + 1)*(2*self.xi - 1)
        self.dN_dxi[:,16] = (1/4)*(1 - self.zeta**2)*self.eta*(self.eta - 1)*(2*self.xi - 1)
        self.dN_dxi[:,17] = (1/4)*(1 - self.zeta**2)*self.eta*(self.eta - 1)*(2*self.xi + 1)
        self.dN_dxi[:,18] = (1/4)*(1 - self.zeta**2)*self.eta*(self.eta + 1)*(2*self.xi + 1)
        self.dN_dxi[:,19] = (1/4)*(1 - self.zeta**2)*self.eta*(self.eta + 1)*(2*self.xi - 1)
        #
        self.dN_dxi[:,20] = (1/2)*(1 - self.eta**2)*self.zeta*(self.zeta - 1)*(-2*self.xi)
        self.dN_dxi[:,21] = (1/2)*(1 - self.eta**2)*self.zeta*(self.zeta + 1)*(-2*self.xi)
        self.dN_dxi[:,22] = (1/2)*(1 - self.zeta**2)*self.eta*(self.eta - 1)*(-2*self.xi)
        self.dN_dxi[:,23] = (1/2)*(1 - self.zeta**2)*self.eta*(self.eta + 1)*(-2*self.xi)
        self.dN_dxi[:,24] = (1/2)*(1 - self.zeta**2)*(1 - self.eta**2)*(2*self.xi - 1)
        self.dN_dxi[:,25] = (1/2)*(1 - self.zeta**2)*(1 - self.eta**2)*(2*self.xi + 1)
        #
        self.dN_dxi[:,26] = (1 - self.zeta**2)*(1 - self.eta**2)*(-2*self.xi)
        #-----------------------------------
        # Calculate derivatives w.r.t. \eta.
        #-----------------------------------
        self.dN_deta       = np.zeros((Parameters.numGauss, Parameters.numGauss), dtype=Parameters.float_dtype)
        #
        self.dN_deta[:,0]  = (1/8)*self.xi*self.zeta*(self.xi - 1)*(self.zeta - 1)*(2*self.eta - 1)
        self.dN_deta[:,1]  = (1/8)*self.xi*self.zeta*(self.xi + 1)*(self.zeta - 1)*(2*self.eta - 1)
        self.dN_deta[:,2]  = (1/8)*self.xi*self.zeta*(self.xi + 1)*(self.zeta - 1)*(2*self.eta + 1)
        self.dN_deta[:,3]  = (1/8)*self.xi*self.zeta*(self.xi - 1)*(self.zeta - 1)*(2*self.eta + 1)
        self.dN_deta[:,4]  = (1/8)*self.xi*self.zeta*(self.xi - 1)*(self.zeta + 1)*(2*self.eta - 1)
        self.dN_deta[:,5]  = (1/8)*self.xi*self.zeta*(self.xi + 1)*(self.zeta + 1)*(2*self.eta - 1)
        self.dN_deta[:,6]  = (1/8)*self.xi*self.zeta*(self.xi + 1)*(self.zeta + 1)*(2*self.eta + 1)
        self.dN_deta[:,7]  = (1/8)*self.xi*self.zeta*(self.xi - 1)*(self.zeta + 1)*(2*self.eta + 1)
        #
        self.dN_deta[:,8]  = (1/4)*(1 - self.xi**2)*self.zeta*(self.zeta - 1)*(2*self.eta - 1)
        self.dN_deta[:,9]  = (1/4)*self.xi*self.zeta*(self.xi + 1)*(self.zeta - 1)*(-2*self.eta)
        self.dN_deta[:,10] = (1/4)*(1 - self.xi**2)*self.zeta*(self.zeta - 1)*(2*self.eta + 1)
        self.dN_deta[:,11] = (1/4)*self.xi*self.zeta*(self.xi - 1)*(self.zeta - 1)*(-2*self.eta)
        self.dN_deta[:,12] = (1/4)*(1 - self.xi**2)*self.zeta*(self.zeta + 1)*(2*self.eta - 1)
        self.dN_deta[:,13] = (1/4)*self.xi*self.zeta*(self.xi + 1)*(self.zeta + 1)*(-2*self.eta)
        self.dN_deta[:,14] = (1/4)*(1 - self.xi**2)*self.zeta*(self.zeta + 1)*(2*self.eta + 1)
        self.dN_deta[:,15] = (1/4)*self.xi*self.zeta*(self.xi - 1)*(self.zeta + 1)*(-2*self.eta)
        self.dN_deta[:,16] = (1/4)*(1 - self.zeta**2)*self.xi*(self.xi - 1)*(2*self.eta - 1)
        self.dN_deta[:,17] = (1/4)*(1 - self.zeta**2)*self.xi*(self.xi + 1)*(2*self.eta - 1)
        self.dN_deta[:,18] = (1/4)*(1 - self.zeta**2)*self.xi*(self.xi + 1)*(2*self.eta + 1)
        self.dN_deta[:,19] = (1/4)*(1 - self.zeta**2)*self.xi*(self.xi - 1)*(2*self.eta + 1)
        #
        self.dN_deta[:,20] = (1/2)*(1 - self.xi**2)*self.zeta*(self.zeta - 1)*(-2*self.eta)
        self.dN_deta[:,21] = (1/2)*(1 - self.xi**2)*self.zeta*(self.zeta + 1)*(-2*self.eta)
        self.dN_deta[:,22] = (1/2)*(1 - self.zeta**2)*(1 - self.xi**2)*(2*self.eta - 1)
        self.dN_deta[:,23] = (1/2)*(1 - self.zeta**2)*(1 - self.xi**2)*(2*self.eta + 1)
        self.dN_deta[:,24] = (1/2)*(1 - self.zeta**2)*self.xi*(self.xi - 1)*(-2*self.eta)
        self.dN_deta[:,25] = (1/2)*(1 - self.zeta**2)*self.xi*(self.xi + 1)*(-2*self.eta)
        #
        self.dN_deta[:,26] = (1 - self.zeta**2)*(1 - self.xi**2)*(-2*self.eta)

    def get_Jacobian(self, Parameters):
        # Compute the element Jacobian.
        self.dx_dxi   = np.einsum('...i, i -> ...', self.dN_dxi,   self.coordinates[:,0], dtype=Parameters.float_dtype)
        self.dx_deta  = np.einsum('...i, i -> ...', self.dN_deta,  self.coordinates[:,0], dtype=Parameters.float_dtype)
        self.dx_dzeta = np.einsum('...i, i -> ...', self.dN_dzeta, self.coordinates[:,0], dtype=Parameters.float_dtype)
        
        self.dy_dxi   = np.einsum('...i, i -> ...', self.dN_dxi,   self.coordinates[:,1], dtype=Parameters.float_dtype)
        self.dy_deta  = np.einsum('...i, i -> ...', self.dN_deta,  self.coordinates[:,1], dtype=Parameters.float_dtype)
        self.dy_dzeta = np.einsum('...i, i -> ...', self.dN_dzeta, self.coordinates[:,1], dtype=Parameters.float_dtype)
        
        self.dz_dxi   = np.einsum('...i, i -> ...', self.dN_dxi,   self.coordinates[:,2], dtype=Parameters.float_dtype)
        self.dz_deta  = np.einsum('...i, i -> ...', self.dN_deta,  self.coordinates[:,2], dtype=Parameters.float_dtype)
        self.dz_dzeta = np.einsum('...i, i -> ...', self.dN_dzeta, self.coordinates[:,2], dtype=Parameters.float_dtype)
                
        self.Je        = np.zeros((Parameters.numGauss,Parameters.numDim,Parameters.numDim), dtype=Parameters.float_dtype)
        self.Je[:,0,0] = self.dx_dxi
        self.Je[:,0,1] = self.dx_deta
        self.Je[:,0,2] = self.dx_dzeta
        self.Je[:,1,0] = self.dy_dxi
        self.Je[:,1,1] = self.dy_deta
        self.Je[:,1,2] = self.dy_dzeta
        self.Je[:,2,0] = self.dz_dxi
        self.Je[:,2,1] = self.dz_deta
        self.Je[:,2,2] = self.dz_dzeta
        
        self.j     = np.zeros(Parameters.numGauss, dtype=Parameters.float_dtype)
        self.Jeinv = np.zeros((Parameters.numGauss,Parameters.numDim,Parameters.numDim), dtype=Parameters.float_dtype)

        for i in range(Parameters.numGauss):
            self.j[i]          = np.linalg.det(self.Je[i,:,:])
            self.Jeinv[i,:,:,] = np.linalg.inv(self.Je[i,:,:])
        
        return

    def get_Jacobian_2D(self, Parameters):
        # Compute the 2D element Jacobian.
        self.dx_dxi_2D  = np.einsum('...i, i -> ...', self.dN_dxi_2D,   self.coordinates[4:8,0], dtype=Parameters.float_dtype)
        self.dx_deta_2D = np.einsum('...i, i -> ...', self.dN_deta_2D,  self.coordinates[4:8,0], dtype=Parameters.float_dtype)
        
        self.dy_dxi_2D  = np.einsum('...i, i -> ...', self.dN_dxi_2D,   self.coordinates[4:8,1], dtype=Parameters.float_dtype)
        self.dy_deta_2D = np.einsum('...i, i -> ...', self.dN_deta_2D,  self.coordinates[4:8,1], dtype=Parameters.float_dtype)
                
        self.Je_2D        = np.zeros((4,2,2), dtype=Parameters.float_dtype)
        self.Je_2D[:,0,0] = self.dx_dxi_2D
        self.Je_2D[:,0,1] = self.dx_deta_2D
        self.Je_2D[:,1,0] = self.dy_dxi_2D
        self.Je_2D[:,1,1] = self.dy_deta_2D
        
        self.j_2D = np.zeros(4, dtype=Parameters.float_dtype)

        for i in range(4):
            self.j_2D[i] = np.linalg.det(self.Je_2D[i,:,:])
        
        return

    def get_Global_DOF(self, a_LM):
        # Set the global degrees of freedom of this element.
        self.DOF    = a_LM[:,self.ID]
        self.numDOF = self.DOF.shape[0]
        return

    def set_Global_Solutions(self, a_D):
        # Set the local solution variables at the current time step.
        self.set_u_global(a_D[self.DOF[0:self.numDOF]])
        return

    def apply_Local_BC(self, a_g):
        # Apply boundary conditions at the element scale.
        if np.any(self.DOF < 0):
            idxs = np.where((self.DOF == -1))[0]
            for idx in idxs:
                self.u_global[idx] = a_g[idx, self.ID]
        return

    def set_u_global(self, a_D):
        # Initialize the global solid displacement (at element level).
        self.u_global = a_D
        return
