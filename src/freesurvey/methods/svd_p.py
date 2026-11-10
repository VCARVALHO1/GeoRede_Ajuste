"""SVD-P: SVD-based Orthogonal Procrustes transformation.

References:
Arun, Huang and Blostein (1987), DOI: 10.1109/TPAMI.1987.4767965.
Umeyama (1991), DOI: 10.1109/34.88573, when scale is estimated.
"""

from freesurvey.Registration_Module import Method1_SequentialSVD


class SVD_P(Method1_SequentialSVD):
    method_code = "SVD-P"
    article_name = "SVD-based Orthogonal Procrustes transformation"


Method1_SequentialSVD_Alias = SVD_P
