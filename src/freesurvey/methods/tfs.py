"""TFS: Total Free Station fixed-scale free-station transformation.

Reference:
Stepien (2019), Total Free Station method for offshore surveying in an oblique
coordinate system.
"""

from freesurvey.Registration_Module import TotalFreeStation_TFS


class TFS(TotalFreeStation_TFS):
    method_code = "TFS"
    article_name = "Total Free Station fixed-scale free-station transformation"


TotalFreeStation_TFS_Alias = TFS
