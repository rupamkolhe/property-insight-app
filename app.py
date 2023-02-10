import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns 
import pandas as pd 
import plotly.graph_objs as go
import streamlit as st 
import json 
import numpy as np 
import pickle
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')

forecast_description = '''
<b style='font-size:18px;'>
Forecasting revenue for properties in Dombivli city for the financial year 2023-2024 on quarterly bases
</b>
'''

predict_description = '''
<b style='font-size:18px;'>
Property prices for buyer, planning to buy property in Dombivli city
</b>
'''

eda_description = '''
<b style='font-size:18px;'>
Explore Property Data set  
</b>
'''

if 'redirect' not in st.session_state:
	st.session_state.redirect = True
if 'page_name' not in st.session_state:
	st.session_state.page_name = ''
if 'inventory' not in st.session_state:
	st.session_state.inventory = []  
if 'rev_page' not in st.session_state:
	st.session_state.rev_page = 0
if 'rev_name' not in st.session_state:
	st.session_state.rev_name = 'std'
if 'sim' not in st.session_state:
	st.session_state.sim = []
if 'bookmark' not in st.session_state:
	st.session_state.bookmark = []
if 'loc_name' not in st.session_state:
	st.session_state.loc_name = None
if 'cust_page' not in st.session_state:
	st.session_state.cust_page = 0

prop_info = pd.read_csv('Prop_info.csv')
raw_data = pd.read_csv('Forecast_property.csv')
pivot = pd.pivot_table(raw_data[['label','name']],index=['label','name']).copy(deep=True)
perm_com = {'LOCALITY':list(pivot.index.get_level_values(1)[pivot.index.get_level_values(0)=='locality'.upper()]),
			'PROJECT':list(pivot.index.get_level_values(1)[pivot.index.get_level_values(0)=='project'.upper()])}
qDecode = pd.DataFrame(pd.to_datetime(raw_data.date).dropna().dt.to_period('Q').drop_duplicates())
qDecode = qDecode.set_index('date').sort_index().resample('Q').interpolate('nearest').index
qDecode = dict(enumerate([str(i.year)+'Q'+str(i.quarter) for i in qDecode]))
qDecode[64] = '2023Q1'
qDecode[65] = '2023Q2'
qDecode[66] = '2023Q3'
qDecode[67] = '2023Q4'
qDecode[68] = '2024Q1'
qEncode = dict(zip(list(qDecode.values()),list(qDecode.keys())))

with open(r'prediction models\featureMap.json','r') as file:
	feature_map = json.load(file)
estModel = pickle.load(open(r'prediction models\estModel.pkl','rb'))
# contains forecasting models 
forecast = {}
filemap = json.load(open(r'forecast models\fileMap.json','rb'))
# here name is in lowercase 
for name in filemap.keys():
	forecast[name] = pickle.load(open(filemap[name],'rb'))

def getData(label='locality',name='Dombivli East',method='nearest'):
	hold_data = raw_data[(raw_data.label==label.upper())&(raw_data.name==name)]
	hold_data.date = pd.to_datetime(hold_data.date)
	hold_data.dropna(inplace=True)
	hold_data = hold_data.drop(columns=['budgetRange','name','label','quarter'])
	hold_data.rename(columns={'price(per.sqft)':'price'},inplace=True)
	hold_data['Quarter'] = hold_data.date.dt.to_period('Q')
	hold_data.drop(columns=['date'],axis=1,inplace=True)
	hold_data.drop_duplicates(inplace=True)
	hold_data = hold_data.groupby(by='Quarter').mean()
	hold_data = hold_data.sort_index()
	hold_data = hold_data.resample('Q').interpolate(method)
	do_forecast = bool(sum([(i.year in [2022,2023]) for i in hold_data.index[-4:]]))
	return {'data':hold_data,'do_forecast':do_forecast}

def filterProp(name:list)->list:
	emp = []
	for n in prop_info.values:
		if (name == n[6]) or (name == n[7]):
			emp.append(list(n))
	return emp

def getKey(name):
	key = ''
	for i in perm_com.keys():
		if name in perm_com[i]:
			key = i
			return key 
def getSuper(x):
	normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
	super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
	res = x.maketrans(''.join(normal), ''.join(super_s))
	return x.translate(res)
def getName(x):
	return x.replace(getSuper('locality'),'').replace(getSuper('project'),'')


def revenueProfile(inv):
	# area,bed_room,floor,unit
	# get rates from forecast 
	unq_name = ([i[0] for i in inv])
	hold = []
	for name in set(unq_name):
		if name == 'Pendse Nagarˡᵒᶜᵃˡᶦᵗʸ':
			res = forecast[getName(name)].forecast(5).iloc[1:]
		else :
			res = forecast[getName(name)].forecast(4)
		x_axis = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in res.index]
		total = 0
		emp = np.zeros(len(res))
		for item in inv:
			if name == item[0]:
				# loop through forecast
				x = []
				for val in res.values.ravel():
					# apply model here 
					eps_carp = 8638.03568957
					eps_floor = -3228.20360704
					eps_br = 42580.48262206
					eps_foCa = 565.27354896
					est_revenue = item[-1]*(eps_carp*item[1]+eps_floor*item[3]+eps_br*item[2]+eps_foCa*val)
					x.append(est_revenue)
				emp += np.array(x)
		hold.append([name, x_axis, list(emp)])
	return hold

def update_default():
	st.session_state.loc_name = st.session_state.loc_n_values

if st.session_state.redirect:
	with st.form('Prediction'):
		st.title('Estimate Property Price')
		st.markdown(predict_description,unsafe_allow_html=True)
		st.image(r'images\customer.jpg')
		if st.form_submit_button('**Visit**'):
			st.session_state.redirect = False
			st.session_state.page_name = 'price_pred'
			st.experimental_rerun()
	with st.form('Forecasting'):
		st.title('Forecast Property Rate') 
		st.markdown(forecast_description,unsafe_allow_html=True)
		st.image(r'images\f_cast.jpg')
		if st.form_submit_button('**Visit**'):
			st.session_state.redirect = False
			st.session_state.page_name = 'forecast'
			st.experimental_rerun()
	with st.form('EDA'):
		st.title('EDA Insights')
		st.markdown(eda_description,unsafe_allow_html=True)
		st.image(r'images\eda.png')
		if st.form_submit_button('**Visit**'):
			st.session_state.redirect = False
			st.session_state.page_name = 'eda'
			st.experimental_rerun()

if st.session_state.page_name == 'forecast':
	# forecasting page 
	if st.sidebar.button('**Home Page**'):
		st.session_state.redirect = True
		st.session_state.page_name = ''
		st.experimental_rerun()


	rate_revenue = option_menu(menu_title=None,
							   options=['Price Rate','Revenue'],
							   icons=['bi-bar-chart','bi-cash-coin'],
						   	   orientation='horizontal')

	if rate_revenue == 'Price Rate':
		loc = st.sidebar.checkbox('LOCALITY',value=True)
		pro = st.sidebar.checkbox('PROJECT')
		value = []
		if loc :
			value += list([i+getSuper('locality') for i in perm_com['locality'.upper()]])
		if pro :
			value += list([i+getSuper('project') for i in perm_com['project'.upper()]])

		nameSelect = st.sidebar.selectbox('Name',tuple(value))
		if nameSelect :
			name = getName(nameSelect)
		else :
			name = ''
		do_forecast = st.sidebar.checkbox('Forecast',value=False)

		compare = st.sidebar.checkbox('compare')
		if compare:
			compareState = False
		else :
			compareState = True

		if name:
			st.title(nameSelect)
			response = getData(label=getKey(name),name=name)
			compWith = st.multiselect('compare with',tuple([x for x in value if x != nameSelect]),disabled=compareState)
			fig = go.Figure()
			emp = []
			x_axis = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in response['data'].index]
			fig.add_trace(go.Scatter(x=x_axis,
									 y=response['data'].values.ravel(),
									 mode='lines',
									 name=nameSelect))
			#  Do forecasting here 
			if do_forecast :
				pred = forecast[name].forecast(4)
				forecast_ax = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in pred.index]
				forecast_ax.insert(0,qEncode[str(response['data'].index[-1].year)+'Q'+str(response['data'].index[-1].quarter)])
				ls = list(pred.values.ravel())
				ls.insert(0,response['data'].values.ravel()[-1])
				fig.add_trace(go.Scatter(x=forecast_ax,
										 y=ls,
										 mode='lines+markers',
										 name='forecast_'+nameSelect,
										 line=dict(color='red')))
				x_axis += forecast_ax
			emp += x_axis
			if compWith:
				for compName in compWith:
					response = getData(label=getKey(getName(compName)),name=getName(compName))
					x_axis = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in response['data'].index]
					emp += x_axis
					fig.add_trace(go.Scatter(x=x_axis,
											 y=response['data'].values.ravel(),
											 mode='lines',
											 name=compName))
					if do_forecast:
						pred = forecast[getName(compName)].forecast(4)
						forecast_ax = [qEncode[str(i.year)+'Q'+str(i.quarter)] for i in pred.index]
						forecast_ax.insert(0,qEncode[str(response['data'].index[-1].year)+'Q'+str(response['data'].index[-1].quarter)])
						ls = list(pred.values.ravel())
						ls.insert(0,response['data'].values.ravel()[-1])
						fig.add_trace(go.Scatter(x=forecast_ax,
												 y=ls,
												 mode='lines+markers',
												 name='forecast_'+compName,
												 line=dict(color='red')))
						x_axis += forecast_ax

			fig.update_traces(marker=dict(size=4))
			fig.update_layout(
					xaxis=dict(
					tickmode='array',
					tickvals=list(set(emp)),
					ticktext=[qDecode[i] for i in set(emp)]
				),
					width=int(800),
					height=int(500),
					margin=dict(
					l=0
					)
			) 
			st.plotly_chart(fig)

	elif rate_revenue == 'Revenue':
		# add revenue prediction model here 
		with st.sidebar.form('add_invent'):
			# names
			value = []
			value += list([i+getSuper('locality') for i in perm_com['locality'.upper()]])
			value += list([i+getSuper('project') for i in perm_com['project'.upper()]])
			name = st.selectbox('Name',options=value)
			# area 
			area = st.number_input('Area',min_value=100.0,max_value=10000.0,step=1.0,value=500.0)
			# bedroom
			bedroom = st.number_input('BedRoom',min_value=1,max_value=4,step=1,value=1)
			# floor 
			floor = st.number_input('Floor',min_value=1,max_value=40,step=1,value=1)
			# units
			units = st.number_input('Units',min_value=1,max_value=100,step=1,value=1)
			if st.form_submit_button('Add'):
				st.session_state.inventory.append([name,area,bedroom,floor,units])
		with st.expander('**Inventory**'):
			st.info('Add Inventory')
			if st.session_state.inventory:
				count = 1
				state = []
				for name,area, bedRoom, floor, qty in st.session_state.inventory:
					with st.form('add_invent'+str(count)):
						col1, col2, col3, col4,col5 = st.columns(5)
						with col1:
							st.subheader(name)
						with col2 :
							st.metric(label='**Carpet Area**',value=area)
						with col3:
							st.metric(label='**Bed Room**',value=bedRoom)
						with col4:
							st.metric(label='**Floor**',value=floor)
						with col5:
							st.metric(label='**Unit**',value=qty)
						remove = st.form_submit_button('**Remove**')
						state.append(int(remove))
						count += 1
				if 1 in state:
					st.write(state.index(1))
					# remove element from list 
					st.session_state.inventory.pop(state.index(1))


					if st.session_state.rev_name == 'std':
						st.session_state.rev_page = 0
						# com -> con
					elif st.session_state.rev_name == 'con':
						st.session_state.rev_page = 1

					st.experimental_rerun()


		# Here add graphs
		with st.sidebar:
			res = option_menu(menu_title=None,
							  options=['Standalone','Consolidated'],
							  orientation='vertical',
							  default_index=st.session_state.rev_page)

		if st.session_state.inventory:
			# standalone OR consolidated 
			if res == 'Standalone':
				st.session_state.rev_name = 'std'
				# for each name(locality/project)
				# model(carpet_area,bedroom,floors,unit) -> total_price
				# plot each total_price
				fig = go.Figure()
				emp = []
				for _name,_ax,_est  in revenueProfile(st.session_state.inventory):
					# draw plot here 
					fig.add_trace(go.Scatter(x=_ax,
											 y=_est,
											 name=_name))
					emp += _ax
				fig.update_traces(marker=dict(size=4))
				fig.update_layout(
					xaxis=dict(
							tickmode='array',
							tickvals=list(set(emp)),
							ticktext=[qDecode[i] for i in set(emp)]
						),
					width=int(700),
					height=int(500),
					margin=dict(l=0),
					title=dict(
							text="<b style='font-size:25px'>Standalone Value of Inventory</b>"
						)
					)
				st.plotly_chart(fig)

			elif res == 'Consolidated':
				st.session_state.rev_name = 'con'
				# model(carpet_area,bedroom,floors,unit) -> total_price
				# plot sum of all total_prices
				rev_res = revenueProfile(st.session_state.inventory)
				axis = []
				for p in rev_res:
					axis += p[1]
				cons_ax = []
				cons_est = []
				nums = sorted(set(axis))
				for x_p in nums:
					i = 0
					cons_ax.append(x_p)
					for _ , new_ax, _val in rev_res:
						zp = dict(zip(new_ax,_val))
						if x_p in zp.keys():
							i += zp[x_p]
					cons_est.append(i)

				fig = go.Figure()
				fig.add_trace(go.Scatter(x=cons_ax,
										 y=cons_est,
										 name='Consolidated Value'))
				fig.update_traces(marker=dict(size=4))
				fig.update_layout(
					xaxis=dict(
							tickmode='array',
							tickvals=list(set(cons_ax)),
							ticktext=[qDecode[i] for i in set(cons_ax)]
						),
					width=int(700),
					height=int(500),
					margin=dict(l=0),
					title=dict(
							text="<b style='font-size:25px'>Consolidated Value of Inventory</b>",
						)
				)
				st.plotly_chart(fig)

elif st.session_state.page_name == 'price_pred':
	# price prediction page
	if st.sidebar.button('Home Page'):
		st.session_state.redirect = True
		st.session_state.page_name = ''
		st.experimental_rerun()
	with st.sidebar:
		opt = option_menu(menu_title=None,
						  options=['Price Est.',f'Bookmark {getSuper(str(len(st.session_state.bookmark)))}'],
						  icons=['fill','fill'],
						  default_index=st.session_state.cust_page)

	if opt == 'Price Est.':
		with st.expander('**Get Estimated Cost**'):
			value = []
			value += list([i+getSuper('locality') for i in perm_com['locality'.upper()]])
			value += list([i+getSuper('project') for i in perm_com['project'.upper()]])

			loc_n = st.multiselect('Locality/Project',options=value,
									key='loc_n_values',
									default=st.session_state.loc_name,
									on_change=update_default)
		
			loc_a = st.number_input('Area',min_value=100.0,max_value=10000.0,step=1.0,value=200.0)
			loc_br = st.number_input('Bed Room',min_value=1,max_value=4,step=1,value=1)
			loc_f = st.number_input('Floor',min_value=1,max_value=40,step=1,value=1)
			# apply model here 
			# get Prediction from model
			st.markdown(f"<b style='font-size:18px;'>Average Estimated Cost : </b>",unsafe_allow_html=True)
			if loc_n:
				# run loop
				for l in loc_n:
					# make predictions here
					empty = [loc_a,loc_f,loc_br]
					empty += [int(i.lower()==getName(l).lower()) for i in feature_map['feature']]
					modelPred = round((np.e**estModel.predict([empty])).ravel()[0],2)

					st.markdown(f"<b style='color:blue;'>{getName(l)} @ ₹{modelPred}</b>",
								unsafe_allow_html=True)
			else :
				st.info('Add Property info in search box to get Average Cost Estimate')

		if loc_n:
			# filter the similar results
			hold = []
			for n_m in loc_n:
				sim_res = filterProp(getName(n_m))[:10]
				if sim_res:
					hold += sim_res
			st.session_state.sim = hold
			if st.session_state.sim:
				#length of result list
				num_result = len(st.session_state.sim)
				col1, col2 = st.columns([6,1])
				with col1:
					st.markdown("<b style='font-size:20px;'>Properties you might find interesting -</b>",
								 unsafe_allow_html=True)
				with col2:
					st.write(f'*results {num_result}*')
				b_state = []
				l_index = []
				for i in range(len(st.session_state.sim)):
					with st.form(f'recom_{i}'):
						col1, col2 = st.columns([1,2])
						with col1:
							st.image(r'images\home.jpg',width=160)
						with col2:
							# locality
							st.write(f'**Locality** : {st.session_state.sim[i][6]}')
							# prop name
							st.write(f'**Property Name** : {st.session_state.sim[i][7]}')
							# carpet area
							st.write(f'**Carpet Area** : {st.session_state.sim[i][0]}')
							# price
							st.write(f'**Price** : {st.session_state.sim[i][2]}')
							# floors 
							st.write(f'**Bedrooms** : {st.session_state.sim[i][-1]}')
							# bedroom
							st.write(f'**Floors** : {st.session_state.sim[i][-2]}')
							if st.session_state.bookmark:
							#	st.write(st.session_state.bookmark)
								parity = False
								for bk in st.session_state.bookmark:
									x = ((bk[6]==st.session_state.sim[i][6])*
										 (bk[7]==st.session_state.sim[i][7])*
										 (bk[0]==st.session_state.sim[i][0])*
										 (bk[2]==st.session_state.sim[i][2])*
										 (bk[-2]==st.session_state.sim[i][-2])*
										 (bk[-1]==st.session_state.sim[i][-1]))
							#		st.write(x)
									parity += x
								if bool(parity):
									label = 'Remove'
								else:
									label = 'Bookmark'
							else : 
								label = 'Bookmark'
							# add label 'Bookmark'
							# if already bookmarked then 
							# change label to 'Remove'
							book_mark = st.form_submit_button(f'**{label}**')
							b_state.append(int(book_mark))
							l_index.append(label)
				if 1 in b_state:
					if l_index[b_state.index(1)] != 'Remove':
						st.session_state.bookmark.append(st.session_state.sim[b_state.index(1)])
						st.session_state.cust_page = 0
						st.experimental_rerun()
					elif l_index[b_state.index(1)] == 'Remove':
						sim_val = st.session_state.sim[b_state.index(1)]
						simInd = []
						parity = False
						for item in st.session_state.bookmark:
							for i in range(len(st.session_state.sim)):
								x = ((item[6]==st.session_state.sim[i][6])*
									 (item[7]==st.session_state.sim[i][7])*
									 (item[0]==st.session_state.sim[i][0])*
									 (item[2]==st.session_state.sim[i][2])*
									 (item[-1]==st.session_state.sim[i][-1])* 
									 (item[-2]==st.session_state.sim[i][-2]))
								parity += x
							simInd.append(int(parity))
						st.session_state.bookmark.pop(simInd.index(1))
						st.session_state.cust_page = 0
						st.experimental_rerun()
					else :
						pass
	else:
		if st.session_state.bookmark:
			rem_book = []
			for i in range(len(st.session_state.bookmark)):
				with st.form(f'bookmark_{i}'):
					col1, col2 = st.columns([1,2])
					with col1:
						st.image(r'images\home.jpg',width=160)
					with col2:
						# locality 
						st.write(f'**Locality** : {st.session_state.bookmark[i][6]}')
						# prop name
						st.write(f'**Property Name** : {st.session_state.bookmark[i][7]}')
						# carpet area
						st.write(f'**Carpet Area** : {st.session_state.bookmark[i][0]}')
						# price
						st.write(f'**Price** : {st.session_state.bookmark[i][2]}')
						# bedroom
						st.write(f'**Bedrooms** : {st.session_state.bookmark[i][-1]}')
						# floors
						st.write(f'**Floors** : {st.session_state.bookmark[i][-2]}')
						rem_b = st.form_submit_button('Remove')
						rem_book.append(int(rem_b))
			if 1 in rem_book:
				st.session_state.bookmark.pop(rem_book.index(1))
				st.session_state.cust_page = 1
				st.experimental_rerun()
		else :
			st.info('No Bookmarks Found')

elif st.session_state.page_name == 'eda':
	# forecasting page 
	if st.sidebar.button('**Home Page**'):
		st.session_state.redirect = True
		st.session_state.page_name = ''
		st.experimental_rerun()
	raw = pd.read_csv('Prop_info.csv').drop(columns=['areaUnit'])
	raw = raw[raw.price != 'price on req']
	raw.price = raw.price.astype('float64')
	with st.expander('**Explore Data**'):
		x_axis = st.selectbox('**X-Axis**',options=[None]+list(raw.columns),index=0)
		y_axis = st.selectbox('**Y-Axis**',options=[None]+list(raw.columns),index=0)
		if x_axis and y_axis:
			if (raw[x_axis].dtypes != 'object') and (raw[y_axis].dtypes != 'object'):
				plots = ['Scatter Plot']
			elif (raw[x_axis].dtypes != 'object') and (raw[y_axis].dtypes == 'object'):
				plots = ['Bar Plot']
			elif (raw[x_axis].dtypes == 'object') and (raw[y_axis].dtypes != 'object'):
				plots = ['Bar Plot']
			elif (raw[x_axis].dtypes == 'object') and (raw[y_axis].dtypes == 'object'):
				plots = ['Bar Plot']

		elif x_axis and not y_axis:
			if raw[x_axis].dtypes != 'object':
				plots = ['Histogram','Box Plot']
			else :
				plots = ['Bar Plot','Pie Plot']
		elif not x_axis and y_axis:
			if raw[y_axis].dtypes != 'object':
				plots = ['Histogram','Box Plot']
			else :
				plots = ['Bar Plot','Pie Plot']
		else :
			plots = []
		if plots :
			disPlot = False
		else :
			disPlot = True

		disp = st.selectbox('**Plots**',options=plots,disabled=disPlot)
		if disp in ['Bar Plot','Pie Plot']:
			lim_dis = False
		else :
			lim_dis = True
		limit = st.selectbox('**Display**',options=['top-10','bottom-10'],
							 disabled=lim_dis)
		plot = st.button('**Plot**')

	if disPlot:
		st.warning('No Plots Available.')
	else :
		if plot :
			# plot here 
			if x_axis and not y_axis:
				if disp == 'Histogram':
					fig = px.histogram(raw,x=[x_axis],title=f'<b>{x_axis}</b>')
					st.plotly_chart(fig)
				elif disp == 'Box Plot':
					fig = px.box(raw,x=[x_axis],title=f'<b>{x_axis}</b>')
					st.plotly_chart(fig)
				elif disp == 'Bar Plot':
					if limit == 'top-10':
						emp = raw[x_axis].value_counts().head()
					elif limit == 'bottom-10':
						emp = raw[x_axis].value_counts().tail()
					fig = px.bar(x=emp.index,y=emp.values)
					st.plotly_chart(fig)
				elif disp == 'Pie Plot':
					if limit == 'top-10':
						emp = raw[x_axis].value_counts().head()
					elif limit == 'bottom-10':
						emp = raw[x_axis].value_counts().tail()
					fig = px.pie(values=emp.values,names=emp.index,title=f'<b>{x_axis}</b>')
					st.plotly_chart(fig)


			elif y_axis and not x_axis:
				if disp == 'Histogram':
					fig = px.histogram(raw,x=[y_axis],title=f'<b>{y_axis}</b>')
					st.plotly_chart(fig)
				elif disp == 'Box Plot':
					fig = px.box(raw,x=[y_axis],title=f'<b>{y_axis}</b>')
					st.plotly_chart(fig)
				elif disp == 'Bar Plot':
					if limit == 'top-10':
						emp = raw[y_axis].value_counts().head()
					elif limit == 'bottom-10':
						emp = raw[y_axis].value_counts().tail()
					fig = px.bar(x=emp.index,y=emp.values)
					st.plotly_chart(fig)
				elif disp == 'Pie Plot':
					if limit == 'top-10':
						emp = raw[y_axis].value_counts().head()
					elif limit == 'bottom-10':
						emp = raw[y_axis].value_counts().tail()
					fig = px.pie(values=emp.values,names=emp.index,title=f'<b>{y_axis}</b>')
					st.plotly_chart(fig)	


			elif x_axis and y_axis:
				if (raw[x_axis].dtypes != 'object') and (raw[y_axis].dtypes != 'object'):
					if disp == 'Scatter Plot':
						fig = px.scatter(raw,x=x_axis,y=y_axis,title=f'{y_axis} Vs {x_axis}')
						st.plotly_chart(fig)
				elif (raw[x_axis].dtypes != 'object') and (raw[y_axis].dtypes == 'object'):
					if disp == 'Bar Plot':
						emp = raw[[y_axis,x_axis]].groupby(by=[y_axis]).mean()
						fig = px.bar(x=emp.values.ravel(),y=emp.index,title=f'{y_axis} Vs mean({x_axis})')
						st.plotly_chart(fig)
				elif (raw[x_axis].dtypes == 'object') and (raw[y_axis].dtypes != 'object'):
					if disp == 'Bar Plot':
						emp = raw[[y_axis,x_axis]].groupby(by=[x_axis]).mean()
						fig = px.bar(x=emp.index,y=emp.values.ravel(),title=f'{y_axis} Vs {x_axis}')
						st.plotly_chart(fig)
				elif (raw[x_axis].dtypes == 'object') and (raw[y_axis].dtypes == 'object'):
					if disp == 'Bar Plot':
						#st.write(raw[[x_axis,y_axis]].pivot_table(index=[x_axis,y_axis]).index)
						raw['dummy'] = np.ones(len(raw))
						emp = raw[[x_axis,y_axis,'dummy']].pivot_table(index=[x_axis,y_axis],values=['dummy'],aggfunc=np.sum)
						emp = emp.reset_index((0,1))

						fig = px.bar(emp,x=x_axis,y='dummy',color=y_axis)
						st.plotly_chart(fig)
						
			else :
				st.warning('No Plots Available.')
		else :
			st.info('Click Plot Button')


