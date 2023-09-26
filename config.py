# configuration for data extraction
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Glaciers_NeurIPS/Glaciers_NeurIPS"
years = ["2013", "2014", "2015", "2016"] #, "2017", "2018", "2019", "2020", "2021", "2022"] 
boundingBox = (77.4879532598204, 31.992646374004327, 77.82063300835557, 32.225292222190056)
extractNDSI = True # False is RGB bands, read documentatiojn if all/different bands of satellite are necessary
clouds = 20 # 0 - 100
extractedCoordinates = [0, 800, 0, 800] # extracted pixels from the satellite images 
allowedMissings = 0.5 # 0 - 1.0 
deltaT = 6 # average data over deltaT months
name = "parvati"


# ECC maximization parameters
stoppingCriterion = 1e-4
maxIterations = 10000

# if patches are extracted 
patchSize = 50
stride = 10
sequenceLength = 4
