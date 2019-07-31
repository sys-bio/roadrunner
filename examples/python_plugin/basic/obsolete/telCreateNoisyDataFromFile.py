import teplugins as tel

p = tel.Plugin ("tel_add_noise")
data = tel.DataSeries.readDataSeries ("testData.dat")

data.plot ()

p.InputData = data
p.Sigma = 0.00005

print 'Sigma is now:' + `p.Sigma`
p.execute()

p.InputData.plot()