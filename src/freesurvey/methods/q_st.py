"""Q-ST: Quasi-Similarity Transformation.

Reference:
Stepien et al. (2020), Sensors 20(22), 6497, DOI: 10.3390/s20226497.
"""

from freesurvey.Registration_Module import Stepien2020_QST


class Q_ST(Stepien2020_QST):
    method_code = "Q-ST"
    article_name = "Quasi-Similarity Transformation"


Stepien2020_QST_Alias = Q_ST
