"""
Centralized repository for sample network traffic data used by the demo UI.
Parses CSV-like rows provided in a fixed column order and exposes helpers
to retrieve random Normal or Attack samples with the exact feature keys
expected by the UI and prediction endpoint.
"""

from __future__ import annotations

import random
from typing import Dict, List, Literal, Optional


# Column order provided by the user
COLUMN_ORDER = [
    'id','dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit',
    'swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean',
    'trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login',
    'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports',
    'attack_cat','label'
]

# Keys expected by the UI and prediction endpoint
EXPECTED_KEYS = [
    'dur','rate','proto','service','state','spkts','dpkts','sbytes','dbytes','sttl','dttl',
    'sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','dwin','stcpb','dtcpb',
    'tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len',
    'ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
    'ct_dst_src_ltm','ct_src_ltm','ct_srv_dst','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd',
    'is_sm_ips_ports'
]


def _to_number(v: str) -> float:
    try:
        if v is None or v == '' or v == '-':
            return 0.0
        # Normalize comma decimals to dot for robustness
        if isinstance(v, str) and ',' in v:
            v = v.replace(',', '.')
        return float(v)
    except Exception:
        return 0.0


def _parse_row(csv_line: str) -> Dict[str, object]:
    parts = [p.strip() for p in csv_line.split(',')]
    # Guard: if commas exist inside values (not in current data), this would need CSV parsing
    data: Dict[str, object] = {}
    for i, col in enumerate(COLUMN_ORDER):
        if i >= len(parts):
            break
        val = parts[i]
        if col in {'proto', 'service', 'state', 'attack_cat'}:
            data[col] = val
        else:
            # numeric fields
            if col == 'label':
                # keep original string label as-is
                data[col] = val
            else:
                data[col] = _to_number(val)
    return data


def _row_to_ui_payload(row: Dict[str, object]) -> Dict[str, object]:
    # Map and subset to exactly the keys expected by the UI/API
    out: Dict[str, object] = {}
    # Copy categorical keys as strings
    out['proto'] = str(row.get('proto', '')).lower()
    out['service'] = str(row.get('service', '-')).lower()
    out['state'] = str(row.get('state', ''))
    # Copy numeric fields
    numeric_map = {
        'dur':'dur','rate':'rate','spkts':'spkts','dpkts':'dpkts','sbytes':'sbytes','dbytes':'dbytes',
        'sttl':'sttl','dttl':'dttl','sload':'sload','dload':'dload','dloss':'dloss','sinpkt':'sinpkt',
        'dinpkt':'dinpkt','sjit':'sjit','djit':'djit','swin':'swin','dwin':'dwin','stcpb':'stcpb',
        'dtcpb':'dtcpb','tcprtt':'tcprtt','synack':'synack','ackdat':'ackdat','smean':'smean',
        'dmean':'dmean','trans_depth':'trans_depth','response_body_len':'response_body_len',
        'ct_srv_src':'ct_srv_src','ct_state_ttl':'ct_state_ttl','ct_dst_ltm':'ct_dst_ltm',
        'ct_src_dport_ltm':'ct_src_dport_ltm','ct_dst_sport_ltm':'ct_dst_sport_ltm',
        'ct_dst_src_ltm':'ct_dst_src_ltm','ct_src_ltm':'ct_src_ltm','ct_srv_dst':'ct_srv_dst',
        'is_ftp_login':'is_ftp_login','ct_ftp_cmd':'ct_ftp_cmd','ct_flw_http_mthd':'ct_flw_http_mthd',
        'is_sm_ips_ports':'is_sm_ips_ports'
    }
    # Add sloss explicitly
    out['sloss'] = row.get('sloss', 0.0)
    for k_src, k_dst in numeric_map.items():
        out[k_dst] = row.get(k_src, 0.0)
    return out


class SampleDataRepository:
    def __init__(self) -> None:
        self._rows: List[Dict[str, object]] = []
        self._normal: List[Dict[str, object]] = []
        self._attack: List[Dict[str, object]] = []
        self._load_rows()

    def _load_rows(self) -> None:
        normal_lines = [
            # Normal
            "1,0.121478,tcp,-,FIN,6,4,258,172,74.08749,252,254,14158.94238,8495.365234,0,0,24.2956,8.375,30.177547,11.830604,255,621772692,2202533631,255,0,0,0,43,43,0,0,1,0,1,1,1,1,0,0,0,1,1,0,Normal,0",
            "2,0.649902,tcp,-,FIN,14,38,734,42014,78.473372,62,252,8395.112305,503571.3125,2,17,49.915,15.432865,61.426934,1387.77833,255,1417884146,3077387971,255,0,0,0,52,1106,0,0,43,1,1,1,1,2,0,0,0,1,6,0,Normal,0",
            "3,1.623129,tcp,-,FIN,8,16,364,13186,14.170161,62,252,1572.271851,60929.23047,1,6,231.875571,102.737203,17179.58686,11420.92623,255,2116150707,2963114973,255,0.111897,0.061458,0.050439,46,824,0,0,7,1,2,1,1,3,0,0,0,2,6,0,Normal,0",
            "4,1.681642,tcp,ftp,FIN,12,12,628,770,13.677108,62,252,2740.178955,3358.62207,1,3,152.876547,90.235726,259.080172,4991.784669,255,1107119177,1047442890,255,0,0,0,52,64,0,0,1,1,2,1,1,3,1,1,0,2,1,0,Normal,0",
            "5,0.449454,tcp,-,FIN,10,6,534,268,33.373826,254,252,8561.499023,3987.059814,2,1,47.750333,75.659602,2415.837634,115.807,255,2436137549,1977154190,255,0.128381,0.071147,0.057234,53,45,0,0,43,1,2,2,1,40,0,0,0,2,39,0,Normal,0",
            "6,0.380537,tcp,-,FIN,10,6,534,268,39.41798,254,252,10112.02539,4709.134766,2,1,39.928778,52.241,2223.730342,82.5505,255,3984155503,1796040391,255,0.172934,0.119331,0.053603,53,45,0,0,43,1,2,2,1,40,0,0,0,2,39,0,Normal,0",
            "7,0.637109,tcp,-,FIN,10,8,534,354,26.683033,254,252,6039.783203,3892.58374,2,1,68.267778,81.137711,4286.82857,119.422719,255,1787309226,1767180493,255,0.143337,0.069136,0.074201,53,44,0,0,43,1,1,1,1,40,0,0,0,1,39,0,Normal,0",
            "8,0.521584,tcp,-,FIN,10,8,534,354,32.593026,254,252,7377.527344,4754.74707,2,1,55.794,66.054141,3770.580726,118.962633,255,205985702,316006300,255,0.116615,0.059195,0.05742,53,44,0,0,43,1,3,3,1,40,0,0,0,3,39,0,Normal,0",
            "9,0.542905,tcp,-,FIN,10,8,534,354,31.313031,254,252,7087.796387,4568.018555,2,1,60.210889,68.109,4060.625597,106.611547,255,884094874,3410317203,255,0.118584,0.066133,0.052451,53,44,0,0,43,1,3,3,1,40,0,0,0,3,39,0,Normal,0",
            "10,0.258687,tcp,-,FIN,10,6,534,268,57.985135,254,252,14875.12012,6927.291016,2,1,27.505111,39.106801,1413.686415,57.200395,255,3368447996,584859215,255,0.087934,0.063116,0.024818,53,45,0,0,43,1,3,3,1,40,0,0,0,3,39,0,Normal,0",
            "11,0.304853,tcp,-,FIN,12,6,4142,268,55.764583,254,252,99641.46875,5878.243164,3,1,25.948818,53.668801,1471.649189,80.404844,255,137150292,2604092885,255,0.097761,0.036508,0.061253,345,45,0,0,11,1,1,1,1,3,0,0,0,1,6,0,Normal,0",
            "12,2.093085,tcp,smtp,FIN,62,28,56329,2212,42.520967,62,252,211825.125,8152.559082,28,8,34.312868,75.092445,3253.278833,106.113453,255,1824722662,860716719,255,0.13114,0.052852,0.078288,909,79,0,0,2,1,1,1,1,2,0,0,0,1,1,0,Normal,0",
            "13,0.416952,tcp,-,FIN,10,6,534,268,35.975363,254,252,9228.879883,4297.856445,2,1,45.088778,64.481199,2610.908343,99.860875,255,88408021,3711983528,255,0.220976,0.094537,0.126439,53,45,0,0,43,1,1,1,1,40,0,0,0,1,39,0,Normal,0",
            "14,0.996221,tcp,-,FIN,10,8,564,354,17.064487,254,252,4079.416016,2489.407471,2,1,110.691222,131.48,6542.815197,202.433047,255,2321780530,2975132930,255,0.169226,0.07516,0.094066,56,44,0,0,11,1,1,1,1,3,0,0,0,2,3,0,Normal,0",
            "15,0.576755,tcp,-,FIN,10,8,534,354,29.475254,254,252,6671.810547,4299.919434,2,1,64.083889,72.27157,4194.544962,116.493234,255,3772251972,4281731981,255,0.113311,0.050849,0.062462,53,44,0,0,43,1,1,1,1,40,0,0,0,1,39,0,Normal,0"
]

        attack_lines = [
            # Attacks
            "273,0.667368,tcp,-,FIN,10,8,564,354,25.473202,254,252,6089.59375,3716.09082,2,1,74.152,93.739141,5450.195183,136.591312,255,2781332383,4013544591,255,0.016439,0.010495,0.005944,56,44,0,0,1,1,1,1,1,1,0,0,0,3,1,0,Reconnaissance,1",
            "274,0.209133,tcp,http,FIN,10,6,508,268,71.724692,254,252,17519.95117,8568.709961,2,1,21.766333,32.622801,1139.187144,51.108875,255,3869127034,1987960674,255,0.066589,0.046011,0.020578,51,45,1,0,12,1,2,2,1,12,0,0,1,2,12,0,Fuzzers,1",
            "275,0.197552,tcp,http,FIN,10,6,758,268,75.929377,254,252,27658.54102,9071.029297,2,1,20.485667,33.139398,1041.396323,50.775418,255,948977096,112926856,255,0.060979,0.031846,0.029133,76,45,1,0,12,1,4,4,1,12,0,0,1,5,12,0,Fuzzers,1",
            "276,0.000008,udp,-,INT,2,0,168,0,125000.0003,254,0,84000000,0,0,0,0.008,0,0,0,0,0,0,0,0,0,0,84,0,0,0,4,2,1,1,1,1,0,0,0,1,1,0,Reconnaissance,1",
            "277,0.483004,tcp,http,FIN,10,8,566,354,35.196396,254,252,8447.134766,5134.533203,2,1,53.667111,58.820855,3414.740225,110.386664,255,2964286253,3984427726,255,0.117386,0.057477,0.059909,57,44,1,0,10,1,3,2,1,10,0,0,1,3,9,0,Fuzzers,1",
            "278,0.523646,tcp,-,FIN,10,6,674,268,28.645306,254,252,9273.44043,3422.15918,2,1,56.548667,91.055602,3256.640996,156.682875,255,2589843549,889491255,255,0.087022,0.06836,0.018662,67,45,0,0,7,1,1,1,1,7,0,0,0,1,7,0,Fuzzers,1",
            "279,2.032866,tcp,ftp,FIN,22,22,1186,1602,21.152403,62,252,4458.729492,6021.056152,7,9,96.803144,94.474999,4838.362529,5599.901392,255,3182323027,3647175709,255,0.087444,0.047985,0.039459,54,73,0,0,1,1,3,1,1,2,1,1,0,2,1,0,Exploits,1",
            "280,0.727527,tcp,ftp-data,FIN,8,8,364,1628,20.617791,62,252,3507.773438,15669.52148,1,2,103.784429,94.951859,6926.842878,177.307719,255,1613063914,4226155936,255,0.106917,0.062859,0.044058,46,204,0,0,1,1,3,1,1,2,0,0,0,2,1,0,Exploits,1",
            "281,0.000009,udp,-,INT,2,0,168,0,111111.1072,254,0,74666664,0,0,0,0.009,0,0,0,0,0,0,0,0,0,0,84,0,0,0,5,2,1,1,1,1,0,0,0,2,1,0,Reconnaissance,1",
            "282,0.681871,tcp,-,FIN,20,8,17266,354,39.596933,254,252,192446.9531,3637.051514,7,1,33.942421,88.76143,2745.442647,108.610156,255,894453596,631719121,255,0.179623,0.060533,0.11909,863,44,0,0,7,1,2,2,1,6,0,0,0,2,6,0,Fuzzers,1",
            "337,1.438237,tcp,http,FIN,10,10,958,2726,13.210619,62,252,4800.321777,13650.0459,2,2,159.804111,152.291438,10857.72385,205.263688,255,750893150,797708189,255,0.141729,0.066968,0.074761,96,273,1,895,2,1,1,1,1,1,0,0,1,1,1,0,DoS,1",
            "360,28.213135,ospf,-,INT,20,0,1280,0,0.673445,254,0,344.803925,0,0,0,1484.90175,0,1831.332375,0,0,0,0,0,0,0,0,0,64,0,0,0,1,2,1,1,1,1,0,0,0,1,1,0,DoS,1",
            "361,28.213135,ospf,-,INT,20,0,1280,0,0.673445,254,0,344.803925,0,0,0,1484.90175,0,1831.332375,0,0,0,0,0,0,0,0,0,64,0,0,0,1,2,1,1,1,1,0,0,0,1,1,0,DoS,1",
            "362,28.213135,ospf,-,INT,20,0,1280,0,0.673445,254,0,344.803925,0,0,0,1484.90175,0,1831.332375,0,0,0,0,0,0,0,0,0,64,0,0,0,1,2,1,1,1,1,0,0,0,1,1,0,DoS,1",
            "442,0.52179,tcp,-,FIN,10,6,684,268,28.747196,254,252,9444.412109,3434.331543,2,1,57.857444,92.87,3218.618558,145.555609,255,1622845501,129146287,255,0.110099,0.057437,0.052662,68,45,0,0,1,1,1,1,1,1,0,0,0,1,1,0,Shellcode,1",
            "448,0.37681,tcp,-,FIN,10,6,562,268,39.807859,254,252,10742.81445,4755.712402,2,1,39.118222,63.509398,1974.184428,82.903047,255,1691428778,2071679324,255,0.087666,0.05926,0.028406,56,45,0,0,2,1,1,1,1,1,0,0,0,1,2,0,Shellcode,1",
            "739,0.000009,kryptolan,-,INT,2,0,200,0,111111.1072,254,0,88888888,0,0,0,0.009,0,0,0,0,0,0,0,0,0,0,100,0,0,0,6,2,4,3,3,6,0,0,0,13,6,0,Analysis,1",
            "809,0.000008,vmtp,-,INT,2,0,200,0,125000.0003,254,0,100000000,0,0,0,0.008,0,0,0,0,0,0,0,0,0,0,100,0,0,0,3,2,2,1,1,4,0,0,0,2,3,0,Analysis,1",
            "957,0.153847,tcp,-,FIN,10,6,732,268,97.499468,254,252,34267.8125,11647.93652,2,1,16.720444,26.663,921.714472,45.354078,255,3835905617,1947290997,255,0.071611,0.020513,0.051098,73,45,0,0,1,1,1,1,1,2,0,0,0,1,1,0,Shellcode,1",
        ]

        for line in normal_lines:
            row = _parse_row(line)
            self._rows.append(row)
            self._normal.append(row)

        for line in attack_lines:
            row = _parse_row(line)
            self._rows.append(row)
            self._attack.append(row)

    def get_random(self, kind: Literal['normal','attack']) -> Dict[str, object]:
        pool = self._normal if kind == 'normal' else self._attack
        if not pool:
            raise ValueError(f"No samples available for kind={kind}")
        row = random.choice(pool)
        return _row_to_ui_payload(row)

    def get_by_id(self, rec_id: int) -> Optional[Dict[str, object]]:
        for r in self._rows:
            if int(r.get('id', -1)) == rec_id:
                return _row_to_ui_payload(r)
        return None


# Singleton repository instance for easy import/use
SAMPLE_REPO = SampleDataRepository()
