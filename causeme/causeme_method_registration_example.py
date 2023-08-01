import json
import bz2
import hashlib

# Filename to save method 
method_dict_file = 'my_method.json.bz2'

# Setup a python dictionary to store descriptions and predictions
method_dict = {}

#################################################
# Method details
#################################################
# Take Vector autoregressive model implemented with a time series model
# from statsmodels as an example (see causeme_method_lib)
# Method name (string without spaces)
method_dict['name'] = "varmodel"
# Longname
method_dict['longname'] = "Vector Autoregressive model"
# Parameters
method_dict['parameters'] = "maxlags"
# Description
method_dict['description'] = "\
Vector Autoregressive model implemented with \
statsmodels.tsa.var.var_model.VAR(data).fit(maxlags) \
statsmodels version = 0.9.0"
# Tags should be chosen to describe the method, you can use the tags on causeme and also others
method_dict['tags'] = "linear, time series"
# Ideally provide a URL of a description paper
method_dict['url_paper'] = "http://conference.scipy.org/proceedings/scipy2010/pdfs/seabold.pdf"
# And even better, provide a URL of code
method_dict['url_code'] = "https://www.statsmodels.org/stable/index.html"

# Provide the SHA1 160-bit of your zipped code if you aim to verify your results
# sha = hashlib.sha1()
# with open('causeme_my_method.zip', "rb") as f:
#     while True:
#         data = f.read(65536)
#         if not data:
#             break
#         sha.update(data)
# method_dict['code_sha'] = sha.hexdigest()


# Save data in your format
print('Writing method dict ...')
method_dict_json = bytes(json.dumps(method_dict), encoding='latin1')
with bz2.BZ2File(method_dict_file, 'w') as mybz2:
    # json.dump(obj=results, fp=mybz2)
    mybz2.write(method_dict_json)