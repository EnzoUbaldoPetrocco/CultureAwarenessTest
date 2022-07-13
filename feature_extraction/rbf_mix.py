#! /usr/bin/env python3

from classificator_featureextraction import SVCClassificator
kernel = [ 'linear', 'rbf' ]
ds_selections = ['chinese', 'french', 'mix']
gaussian_chinese_svc_classificator = SVCClassificator()
gaussian_chinese_svc_classificator.ds_selection = ds_selections[2]
gaussian_chinese_svc_classificator.kernel = kernel[1]
gaussian_chinese_svc_classificator.execute()
