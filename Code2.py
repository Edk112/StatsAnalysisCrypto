#######################################################################################################################
#This code does the following:
#1. Imports data and finds logreturns of daily exchange rates
#2. Runs log likelihood and information criterion tests on each currency for all distributions
#3. plots Q-Q plots
#4. plots P -P plots
#5. Plots Value at Risk graphs and find GoF using KS Test

#Note: To plot graphs, use plt.show(). That and the print commands have been removed or commented for convenience. The code  ight take a long time to run.

#######################################################################################################################
import numpy as np 
import pandas as pd 
import scipy.stats as st 
from matplotlib import pyplot as plt 
import math
from scipy import stats
from scipy.optimize import minimize
import pylab as py
from scipy import optimize
from scipy.special import gamma
from scipy.special import beta

currency = ['Bitcoin', 'Ethereum', 'Dogecoin','Litecoin', 'Nexus', 'Factom', 'PayCoin',  'Phoenixcoin']

def findprice(high, low, price):
	for i in range(len(high)):
		price[i] = (high[i]+low[i])/2

	return price
################################################################################################
f = open('values1.txt', 'w')
f.write("The order of the currencies: ")
print(currency, file = f)
#Getting all the datasets
#BITCOIN
euro_prices = np.genfromtxt('Euro.csv', delimiter = ',')

df = pd.read_csv('Bitcoin.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
bitcoin_dates = df['Date'].tolist()
bitcoin_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))

#ETHEREUM
df = pd.read_csv('Ethereum.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
ethereum_dates = df['Date'].tolist()
ethereum_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))

#DOGECOIN
df = pd.read_csv('Dogecoin.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
dogecoin_dates = df['Date'].tolist()
dogecoin_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))

#LITECOIN
df = pd.read_csv('Litecoin.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
litecoin_dates = df['Date'].tolist()
litecoin_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))

#NEXUS
df = pd.read_csv('Nexus.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
nexus_dates =  df['Date'].tolist()
nexus_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))

#FACTOM
df = pd.read_csv('Factom.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
factom_dates  = df['Date'].tolist()
factom_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))


#PAYCOIN
df = pd.read_csv('Paycoin.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
paycoin_dates = df['Date'].tolist()
paycoin_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))

#PHOENIXCOIN
df = pd.read_csv('Phoenixcoin.csv', usecols = ['Date', 'High', 'Low'])
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
phoenix_dates  = df['Date'].tolist()
phoenix_price = findprice(df['High'].tolist(), df['Low'].tolist(), np.zeros(len(df['High'].tolist())))


###########################################################################################################################################

dates = [bitcoin_dates, ethereum_dates, dogecoin_dates, litecoin_dates, nexus_dates, factom_dates,  paycoin_dates, phoenix_dates]
prices = [bitcoin_price, ethereum_price, dogecoin_price, litecoin_price, nexus_price, factom_price,  paycoin_price, phoenix_price]


def findlog(prices):
	logreturns = []
	for i in range(len(prices)):
		logstuff = np.zeros(len(prices[i]) - 1)
		for j in range(len(logstuff)):
			logstuff[j] = math.log(prices[i][j+1]) - math.log(prices[i][j])
		logreturns.append(logstuff)

	return logreturns

logreturns = findlog(prices)
euro_log = np.zeros(len(euro_prices) - 1)
for i in range(len(euro_log)):
	euro_log[i] = math.log(euro_prices[i+1]) - math.log(euro_prices[i])

distributions = ["Student's T", 'Laplace', 'asymmetric studentsT','normal inverse guassian', 'Beta']

bitcoin_results  = [[], [], [], [], []]
ethereum_results = [[], [], [], [], []]
dogecoin_results = [[], [], [], [], []]
litecoin_results = [[], [], [], [], []]
nexus_results = [[], [], [], [], []]
factom_results = [[], [], [], [], []]
paycoin_results = [[], [], [], [], []]
phoenix_results = [[], [], [], [], []]

results = [bitcoin_results, ethereum_results, dogecoin_results, litecoin_results, nexus_results, factom_results, paycoin_results, phoenix_results]
###################################################################################################################################################

#RUNNING THE TESTS

##########STUDENTS T############################
arraysreturned = ['fit parameters', '-lnL', 'AIC', 'BIC', 'AICc', 'HQC']

count = 0
studentsT = [[], [], [], [], [], []]
#fig4, ax4 = plt.subplots(4, 2, figsize = (8, 8))
#plt.subplots_adjust(hspace = 0.5)
#fig4.suptitle('Student\'s t Distribution')
for i in range(4):
	for j in range(2):
		####fit param#####
		param = stats.t.fit(logreturns[count])
		x = np.linspace(-0.5,0.5,len(logreturns[count]))
		studentsT[0].append(param)
		results[count][0].append(param)

		####-lnL##########
		y = logreturns[count]
		k = len(param)
		N = len(y)
		lnL = (stats.t.logpdf(y ,loc=param[1],scale=param[2],df=param[0]).sum())
		studentsT[1].append((-1)*lnL)
		results[count][0].append((-1)*lnL)

		######aic#########
		aic = (-2)*lnL + 2*len(param)
		studentsT[2].append(aic)
		results[count][0].append(aic)

		######bic########
		bic = (-2)*lnL + 2*np.log(len(y))
		studentsT[3].append(bic)
		results[count][0].append(bic)

		###########AICc#####################
		aicc = aic + (2*k*(k + 1))/(N - k - 1)
		studentsT[4].append(aicc)
		results[count][0].append(aicc)

		############HQC####################
		hqc = -2*lnL + 2*k*np.log(np.log(N))
		studentsT[5].append(hqc)
		results[count][0].append(hqc)

		pdf_fitted = stats.t.pdf(x,loc=param[1],scale=param[2],df=param[0])
		count = count + 1


######LAPLACE###############
count = 0
Laplace = [[], [], [], [], [], []]
#fig5, ax5 = plt.subplots(4, 2, figsize = (8, 8))
#plt.subplots_adjust(hspace = 0.5)
#fig5.suptitle('Laplace')
for i in range(4):
	for j in range(2):
		########fit param##########
		param = stats.laplace.fit(logreturns[count])
		x = np.linspace(-0.5,0.5,len(logreturns[count]))
		Laplace[0].append(param)
		results[count][1].append(param)

		########-lnL###############
		y = logreturns[count]
		k = len(param)
		N = len(y)
		lnL = (stats.laplace.logpdf(y ,loc=param[0],scale=param[1]).sum())
		Laplace[1].append((-1)*lnL)
		results[count][1].append((-1)*lnL)

		###########aic################33
		aic = (-2)*lnL + 2*len(param)
		Laplace[2].append(aic)
		results[count][1].append(aic)

		########bic#############
		bic = (-2)*lnL + 2*np.log(len(y))
		Laplace[3].append(bic)
		results[count][1].append(bic)

		###########AICc#####################
		aicc = aic + (2*k*(k + 1))/(N - k - 1)
		Laplace[4].append(aicc)
		results[count][1].append(aicc)

		############HQC####################
		hqc = -2*lnL + 2*k*np.log(np.log(N))
		Laplace[5].append(hqc)
		results[count][1].append(hqc)

		pdf_fitted = stats.laplace.pdf(x,loc=param[0],scale=param[1])
		count = count + 1



#############asymmetric Students T######################
count = 0
AST = [[], [], [], [], [], []]
#fig6, ax6 = plt.subplots(4, 2, figsize = (8, 8))
#plt.subplots_adjust(hspace = 0.5)
#fig6.suptitle('AST')
for i in range(4):
	for j in range(2):
		###########fit param##########################
		param = stats.nct.fit(logreturns[count])
		x = np.linspace(-0.5,0.5,len(logreturns[count]))
		AST[0].append(param)
		results[count][2].append(param)

		############-lnL#################
		y = logreturns[count]
		k = len(param)
		N = len(y)
		lnL = (stats.nct.logpdf(y ,df = param[0], nc = param[1], loc=param[2],scale=param[3]).sum())
		AST[1].append((-1)*lnL)
		results[count][2].append((-1)*lnL)

		#############AIC###############
		aic = (-2)*lnL + 2*len(param)
		AST[2].append(aic)
		results[count][2].append(aic)

		#############BIC###################
		bic = (-2)*lnL + 2*np.log(len(y))
		AST[3].append(bic)
		results[count][2].append(bic)

		###########AICc#####################
		aicc = aic + (2*k*(k + 1))/(N - k - 1)
		AST[4].append(aicc)
		results[count][2].append(aicc)

		############HQC####################
		hqc = -2*lnL + 2*k*np.log(np.log(N))
		AST[5].append(hqc)
		results[count][2].append(hqc)

		pdf_fitted = stats.nct.pdf(x,df = param[0], nc = param[1], loc=param[2],scale=param[3])
		count = count + 1


##############normal inverse gauss########################
count = 0
NIG = [[], [], [], [], [], []]

LLH = []
#fig7, ax7 = plt.subplots(4, 2, figsize = (8, 8))
#plt.subplots_adjust(hspace = 0.5)
#fig7.suptitle('Normal Inverse gauss')
for i in range(4):
	for j in range(2):
		#######fit_param###########
		param = stats.norminvgauss.fit(logreturns[count])
		x = np.linspace(-0.5,0.5,len(logreturns[count]))
		NIG[0].append(param)
		results[count][3].append(param)

		############-lnL#################
		y = logreturns[count]
		k = len(param)
		N = len(y)
		lnL = (stats.norminvgauss.logpdf(y ,a = param[0], b=param[1], loc=param[2],scale=param[3]).sum())
		NIG[1].append((-1)*lnL)
		results[count][3].append((-1)*lnL)
		
		#############BIC###################
		bic = (-2)*lnL + 2*np.log(len(y))
		NIG[3].append(bic)
		results[count][3].append(bic)

		#############AIC################
		aic = (-2)*lnL + 2*len(param)
		NIG[2].append(aic)
		results[count][3].append(aic)

		###########AICc#####################
		aicc = aic + (2*k*(k + 1))/(N - k - 1)
		NIG[4].append(aicc)
		results[count][3].append(aicc)

		############HQC####################
		hqc = -2*lnL + 2*k*np.log(np.log(N))
		NIG[5].append(hqc)
		results[count][3].append(hqc)

		pdf_fitted = stats.norminvgauss.pdf(x,a = param[0], b=param[1], loc=param[2],scale=param[3])
		
		count = count + 1


#print(NIG[5])
#plt.show()
##########################Beta#############################
count = 0
BD = [[], [], [], [], [], []]

LLH = []
#fig7, ax7 = plt.subplots(4, 2, figsize = (8, 8))
#plt.subplots_adjust(hspace = 0.5)
#fig7.suptitle('Normal Inverse gauss')
for i in range(4):
	for j in range(2):
		#######fit_param###########
		param = stats.beta.fit(logreturns[count])
		x = np.linspace(-0.5,0.5,len(logreturns[count]))
		BD[0].append(param)
		results[count][4].append(param)

		############-lnL#################
		y = logreturns[count]
		k = len(param)
		N = len(y)
		lnL = (stats.beta.logpdf(y ,a = param[0], b = param[1], loc=param[2],scale=param[3]).sum())
		BD[1].append((-1)*lnL)
		results[count][4].append((-1)*lnL)
		
		#############BIC###################
		bic = (-2)*lnL + 2*np.log(len(y))
		BD[3].append(bic)
		results[count][4].append(bic)

		#############AIC################
		aic = (-2)*lnL + 2*len(param)
		BD[2].append(aic)
		results[count][4].append(aic)

		###########AICc#####################
		aicc = aic + (2*k*(k + 1))/(N - k - 1)
		BD[4].append(aicc)
		results[count][4].append(aicc)

		############HQC####################
		hqc = -2*lnL + 2*k*np.log(np.log(N))
		BD[5].append(hqc)
		results[count][4].append(hqc)

		pdf_fitted = stats.norminvgauss.pdf(x,a = param[0], b=param[1], loc=param[2],scale=param[3])
		
		count = count + 1


#PRINTING TEST RESULTS TO A FILE
#print(results)

f = open('values2.txt', 'w')
for i in range(len(currency)):
	print(currency[i], file = f)
	print("\n", file = f)
	print(results[i], file = f)
	print("\n", file = f)


ks = []
##################################VaR plots and KS test##########################################
ks = []
#############Bitcoin#########
param =  [-0.002364249500708875, 0.021924764875966056]
q = np.linspace(0, 1, 100)

e_param = stats.laplace.fit(euro_log)
VaR = stats.laplace(param[0], param[1]).ppf(q)
e_var = stats.laplace(e_param[0], e_param[1]).ppf(q)
### test#####
d, p = stats.kstest(logreturns[0], cdf = 'laplace', args = (param[0], param[1]))

ks.append(p)

#plt.plot(q, VaR, 'r-', label = 'Bitcoin')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()

##########Ethereum##################
param = [-0.0013526135404853836, 0.035713019636230386]
q = np.linspace(0, 1, 100)
e_param = stats.laplace.fit(euro_log)
VaR = stats.laplace(param[0], param[1]).ppf(q)
e_var = stats.laplace(e_param[0], e_param[1]).ppf(q)
d, p = stats.kstest(logreturns[1], cdf = 'laplace', args = (param[0], param[1]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Ethereum')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()


###############DOGECOIN################
param = [0.11944945909723138, -0.026530755256677835, 0.0024121276834745383, 0.01998674804467044]
q = np.linspace(0, 1, 100)
e_param = stats.norminvgauss.fit(euro_log)
VaR = stats.norminvgauss(param[0], param[1], param[2], param[3]).ppf(q)
e_var = stats.norminvgauss(e_param[0], e_param[1], e_param[2], e_param[3]).ppf(q)
d, p = stats.kstest(logreturns[2], cdf = 'norminvgauss', args = (param[0], param[1], param[2], param[3]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Dogecoin')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()

################LITECOIN###############
param = [1.8693373494312953, -0.08195764686173776, 0.0017432733966738605, 0.021626187586807188]
q = np.linspace(0, 1, 100)
e_param = stats.nct.fit(euro_log)
VaR = stats.nct(param[0], param[1], param[2], param[3]).ppf(q)
e_var = stats.nct(e_param[0], e_param[1], e_param[2], e_param[3]).ppf(q)
d, p = stats.kstest(logreturns[3], cdf = 'nct', args = (param[0], param[1], param[2], param[3]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Litecoin')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()

################NEXUS################
param = [2.269163744926829, -0.22915037600364918, 0.014003933425093236, 0.048491139607216696]
q = np.linspace(0, 1, 100)
e_param = stats.nct.fit(euro_log)
VaR = stats.nct(param[0], param[1], param[2], param[3]).ppf(q)
e_var = stats.nct(e_param[0], e_param[1], e_param[2], e_param[3]).ppf(q)
d, p = stats.kstest(logreturns[4], cdf = 'nct', args = (param[0], param[1], param[2], param[3]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Nexus')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()

#########################FACTOM################
param = [0.4391782758808524, -0.05533105941441145, 0.003687767421713491, 0.047319765631733554]
q = np.linspace(0, 1, 100)
e_param = stats.norminvgauss.fit(euro_log)
VaR = stats.norminvgauss(param[0], param[1], param[2], param[3]).ppf(q)
e_var = stats.norminvgauss(e_param[0], e_param[1], e_param[2], e_param[3]).ppf(q)
d, p = stats.kstest(logreturns[5], cdf = 'norminvgauss', args = (param[0], param[1], param[2], param[3]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Factom')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()

###################PAYCOIN###################
param = [0.11886656344862934, -0.006579353951755887, 0.00342072907819126, 0.062987333793776]
q = np.linspace(0, 1, 100)
e_param = stats.norminvgauss.fit(euro_log)
VaR = stats.norminvgauss(param[0], param[1], param[2], param[3]).ppf(q)
e_var = stats.norminvgauss(e_param[0], e_param[1], e_param[2], e_param[3]).ppf(q)
d, p = stats.kstest(logreturns[6], cdf = 'norminvgauss', args = (param[0], param[1], param[2], param[3]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Paycoin')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()

##########PHOENIX##########################
param = [0.2073531294955301, -0.03720650185250752, 0.00919009772713967, 0.0618062035245221]
q = np.linspace(0, 1, 100)
e_param = stats.norminvgauss.fit(euro_log)
VaR = stats.norminvgauss(param[0], param[1], param[2], param[3]).ppf(q)
e_var = stats.norminvgauss(e_param[0], e_param[1], e_param[2], e_param[3]).ppf(q)
d, p = stats.kstest(logreturns[7], cdf = 'norminvgauss', args = (param[0], param[1], param[2], param[3]))

ks.append(p)
#plt.plot(q, VaR, 'r-', label = 'Phoenixcoin')
#plt.plot(q, e_var, 'b--', label = 'Euro')
#plt.legend()


######################################### PP Plots#############################
import seaborn as sns
def pp_plot(x, dist, title):
	
	n = len(x)
	p = np.arange(1, n + 1) / n - 0.5 / n
	pp = np.sort(dist.cdf(x))
	plt.figure()
	sns.scatterplot(x=p, y=pp, color='blue', edgecolor='blue')
	plt.suptitle(title)
	plt.xlabel('Theoretical Probabilities')
	plt.ylabel('Sample Probabilities')
	plt.margins(x=0, y=0)
	plt.plot(np.linspace(0, 1), np.linspace(0, 1), 'r', lw=2)

	#plt.show()

param =  [-0.002364249500708875, 0.021924764875966056]
pp_plot(logreturns[0], stats.laplace(param[0], param[1]), 'Bitcoin')

param = [-0.0013526135404853836, 0.035713019636230386]
pp_plot(logreturns[1], stats.laplace(param[0], param[1]), 'Ethereum')

param = [0.11944945909723138, -0.026530755256677835, 0.0024121276834745383, 0.01998674804467044]
pp_plot(logreturns[2], stats.norminvgauss(param[0], param[1], param[2], param[3]), 'Dogecoin')

param = [1.8693373494312953, -0.08195764686173776, 0.0017432733966738605, 0.021626187586807188]
pp_plot(logreturns[3], stats.nct(param[0], param[1], param[2], param[3]), 'Litecoin')

param = [2.269163744926829, -0.22915037600364918, 0.014003933425093236, 0.048491139607216696]
pp_plot(logreturns[4], stats.nct(param[0], param[1], param[2], param[3]), 'Nexus')

param = [0.4391782758808524, -0.05533105941441145, 0.003687767421713491, 0.047319765631733554]
pp_plot(logreturns[5], stats.norminvgauss(param[0], param[1], param[2], param[3]), 'Factom')

param = [0.11886656344862934, -0.006579353951755887, 0.00342072907819126, 0.062987333793776]
pp_plot(logreturns[6], stats.norminvgauss(param[0], param[1], param[2], param[3]), 'PayCoin')

param = [0.2073531294955301, -0.03720650185250752, 0.00919009772713967, 0.0618062035245221]
pp_plot(logreturns[7], stats.norminvgauss(param[0], param[1], param[2], param[3]), 'Phoenixcoin')

################################QQ Plots##########################################
import pylab 
import scipy.stats as stats
#####################BITCOIN####################
param =  [-0.002364249500708875, 0.021924764875966056]
#stats.probplot(logreturns[0], dist="laplace", sparams = param,  plot=pylab)
pylab.title('Bitcoin')


####################ETHEREUM######################
param = [-0.0013526135404853836, 0.035713019636230386]
#stats.probplot(logreturns[1], dist="laplace", plot=pylab)
pylab.title('Ethereum')

####################DOGECOIN#######################
param = [0.11944945909723138, -0.026530755256677835, 0.0024121276834745383, 0.01998674804467044]
#stats.probplot(logreturns[2], dist= "norminvgauss", sparams = param , plot=pylab)
pylab.title('Dogecoin')

####################LITECOIN########################
param = [1.8693373494312953, -0.08195764686173776, 0.0017432733966738605, 0.021626187586807188]
#stats.probplot(logreturns[3], dist="nct", sparams = param,  plot=pylab)
pylab.title('Litecoin')

###################NEXUS########################
param = [2.269163744926829, -0.22915037600364918, 0.014003933425093236, 0.048491139607216696]
#stats.probplot(logreturns[4], dist="nct", sparams = param,  plot=pylab)
pylab.title('Nexus')

######################FACTOM#############################
param = [0.4391782758808524, -0.05533105941441145, 0.003687767421713491, 0.047319765631733554]
#stats.probplot(logreturns[5], dist="norminvgauss", sparams = param,  plot=pylab)
pylab.title('Factom')

##################PAYCOIN#############################
param = [0.11886656344862934, -0.006579353951755887, 0.00342072907819126, 0.062987333793776]
stats.probplot(logreturns[6], dist="norminvgauss", sparams = param,  plot=pylab)
pylab.title('Paycoin')

#################PHOENIX#################################
param = [0.2073531294955301, -0.03720650185250752, 0.00919009772713967, 0.0618062035245221]
stats.probplot(logreturns[7], dist="norminvgauss", sparams = param,  plot=pylab)
pylab.title('Phoenixcoin')
#pylab.show()

