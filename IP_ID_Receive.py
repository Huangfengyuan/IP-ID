import fcntl
import collections

from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6PacketTooBig
from scapy.sendrecv import send, sniff, sr, sr1


file_no =9
f_name = './result3/R_%d.txt'
f = open(f_name % file_no, 'a+', encoding='utf-8')

DICT=collections.OrderedDict()

def IPID_Write(packet):
    global f,DICT,count
    try:
        seq=packet['ICMPv6 Echo Reply'].seq
        if seq == 0:
            pass
        else:
            src=packet['IPv6'].src
            if 'IPv6 Extension Header - Fragmentation header' in packet:
                ipid=packet['IPv6 Extension Header - Fragmentation header'].id
            else:
                ipid=0

            if src in DICT:
                DICT[src][seq-1]=ipid
            else:
                DICT[src]=[-1]*30
                DICT[src][seq-1]=ipid
                
                if len(DICT)>=400:
                    tmp=DICT.popitem(last=False)
                    str_f=tmp[0] +' [' +' '.join([str(i) for i in tmp[1]]) + ']'
                    fcntl.flock(f, fcntl.LOCK_EX)
                    print(str_f, file=f)          #写入 'ipv6_address [ipid]' 丢包部分以-1代替
                    fcntl.flock(f, fcntl.LOCK_UN)
    except:
        pass
                
source_A=''
source_B=''

sniff(filter='dst host %s or %s and icmp6'%(source_A,source_B),prn=IPID_Write,iface='',store=0)
while DICT:
    tmp=DICT.popitem(last=False)
    str_f=tmp[0] +' [' +' '.join([str(i) for i in tmp[1]]) + ']'
    fcntl.flock(f, fcntl.LOCK_EX)
    print(str_f, file=f)
    fcntl.flock(f, fcntl.LOCK_UN)
f.close()
