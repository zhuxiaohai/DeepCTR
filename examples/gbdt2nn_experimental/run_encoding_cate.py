import os
import pandas as pd
from deepctr.models.gbdt2nn.encoding_cate import CateEncoder


def online_encoding():
    ec.fit_transform(pd.read_csv(train_csv_path), out_dir)
    data_type = ['2test', '3oot', '4oot', '5oot']
    for i in range(len(data_type)):
        df = pd.read_csv(test_csv_path)
        ec.transform(df[df['set'] == data_type[i]], out_dir + '/{}_'.format(data_type[i]))


num_bins = 10
threshold = 10
thresrate = 0.99
numI = 13
numC = 26
train_csv_path = '../data/gbdt2nn/train.csv'
test_csv_path = '../data/gbdt2nn/test.csv'
out_dir = '../data/gbdt2nn/risk_offline_cate/'
online = True

# for criteo
# cate_col = ['C'+str(i) for i in range(1, args['numC']+1)]
# nume_col = ['I'+str(i) for i in range(1, args['numI']+1)]
# label_col = 'Label'

# for flight delay
# cate_col = ["Month_cate", "DayofMonth_cate", "DayOfWeek_cate", "DepTime_cate", "UniqueCarrier", "Origin", "Dest"]
# nume_col = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"]
# label_col = 'dep_delayed_15min'

# for bike demand
# cate_col = ['month_cate', 'day_cate', 'hour_cate', 'dayofweek_cate', 'season', 'weather_cate']
# nume_col = ['month', 'day', 'hour', 'dayofweek', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']
# label_col = 'count'

# for yahoo
# nume_col = ['f_'+str(idx) for idx in range(699)]
# cate_col = []
# label_col='Label'

# for talking
# nume_col = ['click_hour']
# cate_col = ['ip','app','device','os','channel','click_hour_cate']
# label_col='is_attributed'

# for zillow
# nume_col = ['bathroomcnt','bedroomcnt','calculatedbathnbr','threequarterbathnbr','finishedfloor1squarefeet','calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','fireplacecnt','fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet','numberofstories','poolcnt','poolsizesum','roomcnt','unitcnt','yardbuildingsqft17','yardbuildingsqft17','taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','taxdelinquencyyear','yearbuilt']
# cate_col = ['architecturalstyletypeid', 'yearbuilt_cate', 'buildingqualitytypeid', 'propertyzoningdesc', 'regionidneighborhood', 'yardbuildingsqft26', 'fireplaceflag', 'propertycountylandusecode', 'hashottuborspa', 'basementsqft', 'fips', 'buildingclasstypeid', 'pooltypeid2', 'pooltypeid10', 'regionidcounty', 'heatingorsystemtypeid', 'rawcensustractandblock', 'censustractandblock', 'taxdelinquencyflag', 'airconditioningtypeid', 'pooltypeid7', 'regionidcity', 'regionidzip', 'decktypeid', 'typeconstructiontypeid', 'propertylandusetypeid', 'storytypeid']
# label_col = 'logerror'

# for malware
# nume_col = ['AVProductsInstalled', 'AVProductsEnabled','Census_ProcessorCoreCount','Census_PrimaryDiskTotalCapacity','Census_SystemVolumeTotalCapacity','Census_TotalPhysicalRAM','Census_InternalPrimaryDiagonalDisplaySizeInInches','Census_InternalPrimaryDisplayResolutionHorizontal','Census_InternalPrimaryDisplayResolutionVertical','Census_InternalBatteryNumberOfCharges','Census_OSBuildNumber','Census_OSBuildRevision']
# cate_col = ['IeVerIdentifier', 'Census_ProcessorClass', 'Processor', 'Census_OEMNameIdentifier', 'Firewall', 'Census_FirmwareVersionIdentifier', 'AppVersion', 'CityIdentifier', 'Census_PowerPlatformRoleName', 'Census_OSBranch', 'AvSigVersion', 'Census_IsPortableOperatingSystem', 'Census_OSEdition', 'Census_GenuineStateName', 'OsVer', 'Census_IsAlwaysOnAlwaysConnectedCapable', 'HasTpm', 'Census_IsWIMBootEnabled', 'Census_IsFlightsDisabled', 'Census_IsFlightingInternal', 'AutoSampleOptIn', 'SkuEdition', 'SMode', 'Census_OSWUAutoUpdateOptionsName', 'Wdft_IsGamer', 'Census_OSUILocaleIdentifier', 'Census_IsPenCapable', 'OsPlatformSubRelease', 'Census_IsTouchEnabled', 'IsBeta', 'Census_HasOpticalDiskDrive', 'SmartScreen', 'IsProtected', 'Census_ProcessorModelIdentifier', 'Census_PrimaryDiskTypeName', 'OrganizationIdentifier', 'Census_ActivationChannel', 'Census_IsSecureBootEnabled', 'Census_OSArchitecture', 'CountryIdentifier', 'Census_ThresholdOptIn', 'Census_ChassisTypeName', 'Census_OSSkuName', 'Census_FirmwareManufacturerIdentifier', 'PuaMode', 'Census_MDC2FormFactor', 'ProductName', 'AVProductStatesIdentifier', 'GeoNameIdentifier', 'Census_OSInstallLanguageIdentifier', 'Census_ProcessorManufacturerIdentifier', 'Census_IsVirtualDevice', 'UacLuaenable', 'Census_OSInstallTypeName', 'Platform', 'Census_DeviceFamily', 'Census_InternalBatteryType', 'RtpStateBitfield', 'DefaultBrowsersIdentifier', 'OsBuild', 'OsSuite', 'EngineVersion', 'Census_FlightRing', 'IsSxsPassiveMode', 'Census_OSVersion', 'Wdft_RegionIdentifier', 'LocaleEnglishNameIdentifier', 'Census_OEMModelIdentifier', 'OsBuildLab']
# label_col = 'HasDetections'

# for nips_a
# nume_col = ['13', '14', '15', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '41', '42', '43', '44', '45', '46', '47', '48', '49', '54']
# cate_col = ['0', '2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '16', '17', '18', '27', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '50', '51', '52', '53', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '72', '74', '75', '78', '80', '81']
# label_col = 'label'

# for nips_d
# nume_col = ['3', '5', '6', '7', '11', '12', '16', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '46', '49', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '70', '71', '72', '73', '74', '75']
# cate_col = ['0', '1', '2', '4', '8', '10', '13', '14', '15', '17', '18', '39', '40', '43', '44', '45', '47', '69']
# label_col = 'label'

# for nips b
# nume_col = ['9','11','12','13','14','15','16']
# cate_col = ['0','1','2','3','4','6','7','8','10','17','18','19','20','21','22','23','24']
# label_col = 'label'

# for risk experiment
nume_col = ['ali_rain_score', 'bj_jc_m36_consume_cnt', 'td_zhixin_score', 'hds_36m_purchase_steady',
            'hds_36m_total_purchase_cnt', 'hds_36m_month_max_purchase_money_excp_doub11_12',
            'hds_36m_doub11_12_total_purchase_money', 'ab_local_ratio', 'ab_mobile_cnt', 'cust_id_area',
            'cust_work_city', 'immediate_relation_cnt', 'relation_contact_cnt', 'study_app_cnt',
            'ab_local_cnt', 'ab_prov_cnt', 'credit_repayment_score_bj_2', 'td_xyf_dq_score']
cate_col = ['hds_phone_rich_rank', 'hds_mobile_rich', 'hds_recent_consumme_active_rank', 'idcard_district_grade',
            'idcard_rural_flag', 'selffill_degree', 'selffill_is_have_creditcard', 'selffill_marital_status',
            'hds_mobile_reli_rank_Ma', 'hds_mobile_reli_rank_Mb', 'hds_mobile_reli_rank_M0', 'is_ios', 'is_male']
label_col = 'fpd4'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
ec = CateEncoder(cate_col, nume_col, threshold, thresrate, num_bins, label_col)

if online:
    online_encoding()
else:
    ec.fit_transform(pd.read_csv(train_csv_path), out_dir)
    ec.transform(pd.read_csv(test_csv_path), out_dir)
