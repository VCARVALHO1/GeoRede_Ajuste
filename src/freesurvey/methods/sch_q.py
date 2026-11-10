"""SCH-Q: Schut quaternion-based spatial similarity transformation.

References:
Schut (1973) and Horn (1987), DOI: 10.1364/JOSAA.4.000629.
"""

from freesurvey.Registration_Module import Schut1973_SpaceTime


class SCH_Q(Schut1973_SpaceTime):
    method_code = "SCH-Q"
    article_name = "Schut quaternion-based spatial similarity transformation"


Schut1973_SpaceTime_Alias = SCH_Q
