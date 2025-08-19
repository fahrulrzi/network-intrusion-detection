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
    'sload','dload','dloss','sinpkt','dinpkt','sjit','djit','swin','dwin','stcpb','dtcpb',
    'tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len',
    'ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
    'ct_dst_src_ltm','ct_src_ltm','ct_srv_dst','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd',
    'is_sm_ips_ports'
]


def _to_number(v: str) -> float:
    try:
        if v is None or v == '' or v == '-':
            return 0.0
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
    out['service'] = str(row.get('service', '-'))
    out['state'] = str(row.get('state', '')).lower()
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
            "1,0.000011,udp,-,INT,2,0,496,0,90909.0902,254,0,180363632,0,0,0,0.011,0,0,0,0,0,0,0,0,0,0,248,0,0,0,2,2,1,1,1,2,0,0,0,1,2,0,Normal,0",
            "2,0.000008,udp,-,INT,2,0,1762,0,125000.0003,254,0,881000000,0,0,0,0.008,0,0,0,0,0,0,0,0,0,0,881,0,0,0,2,2,1,1,1,2,0,0,0,1,2,0,Normal,0",
            "3,0.000005,udp,-,INT,2,0,1068,0,200000.0051,254,0,854400000,0,0,0,0.005,0,0,0,0,0,0,0,0,0,0,534,0,0,0,3,2,1,1,1,3,0,0,0,1,3,0,Normal,0",
            "4,0.000006,udp,-,INT,2,0,900,0,166666.6608,254,0,600000000,0,0,0,0.006,0,0,0,0,0,0,0,0,0,0,450,0,0,0,3,2,2,2,1,3,0,0,0,2,3,0,Normal,0",
            "5,0.00001,udp,-,INT,2,0,2126,0,100000.0025,254,0,850400000,0,0,0,0.01,0,0,0,0,0,0,0,0,0,0,1063,0,0,0,3,2,2,2,1,3,0,0,0,2,3,0,Normal,0",
            "6,0.000003,udp,-,INT,2,0,784,0,333333.3215,254,0,1045333312,0,0,0,0.003,0,0,0,0,0,0,0,0,0,0,392,0,0,0,2,2,2,2,1,2,0,0,0,2,2,0,Normal,0",
            "7,0.000006,udp,-,INT,2,0,1960,0,166666.6608,254,0,1306666624,0,0,0,0.006,0,0,0,0,0,0,0,0,0,0,980,0,0,0,2,2,2,2,1,2,0,0,0,2,2,0,Normal,0",
            "8,0.000028,udp,-,INT,2,0,1384,0,35714.28522,254,0,197714288,0,0,0,0.028,0,0,0,0,0,0,0,0,0,0,692,0,0,0,3,2,1,1,1,3,0,0,0,1,3,0,Normal,0",
            "9,0,arp,-,INT,1,0,46,0,0,0,0,0,0,0,0,60000.688,0,0,0,0,0,0,0,0,0,0,46,0,0,0,2,2,2,2,2,2,0,0,0,2,2,1,Normal,0",
            "10,0,arp,-,INT,1,0,46,0,0,0,0,0,0,0,0,60000.712,0,0,0,0,0,0,0,0,0,0,46,0,0,0,2,2,2,2,2,2,0,0,0,2,2,1,Normal,0",
            "11,0,arp,-,INT,1,0,46,0,0,0,0,0,0,0,0,60000.688,0,0,0,0,0,0,0,0,0,0,46,0,0,0,2,2,2,2,2,2,0,0,0,2,2,1,Normal,0",
            "12,0,arp,-,INT,1,0,46,0,0,0,0,0,0,0,0,60000.712,0,0,0,0,0,0,0,0,0,0,46,0,0,0,2,2,2,2,2,2,0,0,0,2,2,1,Normal,0",
            "13,0.000004,udp,-,INT,2,0,1454,0,250000.0006,254,0,1454000000,0,0,0,0.004,0,0,0,0,0,0,0,0,0,0,727,0,0,0,3,2,1,1,1,3,0,0,0,1,3,0,Normal,0",
            "14,0.000007,udp,-,INT,2,0,2062,0,142857.1409,254,0,1178285696,0,0,0,0.007,0,0,0,0,0,0,0,0,0,0,1031,0,0,0,3,2,1,1,1,3,0,0,0,1,3,0,Normal,0",
            "15,0.000011,udp,-,INT,2,0,2040,0,90909.0902,254,0,741818176,0,0,0,0.011,0,0,0,0,0,0,0,0,0,0,1020,0,0,0,1,2,1,1,1,1,0,0,0,1,1,0,Normal,0",
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
