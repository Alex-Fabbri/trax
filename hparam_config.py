import sys
import gin

config_fname = sys.argv[1]
config_outfname = sys.argv[2]

config = gin.parse_config_file(config_fname)
key = 
while True:
    key = raw_input("\n Enter a parameter to change. Type break to exit")
    if key == "break":
        break
    value = raw_input("\n Enter a value for the parameter: " key + ". Type break to exit")
    if value == "break":
        break
    gin.bind_parameter(key, value)

with open(config_outfname, 'w') as configfile:
    configfile.write(gin.config_str())

