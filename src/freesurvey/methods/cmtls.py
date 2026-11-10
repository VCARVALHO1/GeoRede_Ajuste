"""CMTLS: Constrained Multivariate Total Least Squares.

Reference:
Tao, Hua and Feng (2020), DOI: 10.1590/S1982-21702020000400021.
"""

from freesurvey.Registration_Module import Tao2020_CMTLS_Rigorous


class CMTLS(Tao2020_CMTLS_Rigorous):
    method_code = "CMTLS"
    article_name = "Constrained Multivariate Total Least Squares for 3D similarity transformation"


Tao2020_CMTLS_Rigorous_Alias = CMTLS
