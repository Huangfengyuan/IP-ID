import fcntl
import collections

from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6PacketTooBig
from scapy.sendrecv import send, sniff, sr, sr1


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
                DICT[src]=[-1]*2*rounds
                DICT[src][seq-1]=ipid

                if len(DICT)>=1000:
                    tmp=DICT.popitem(last=False)
                    str_f=tmp[0] +' [' +' '.join([str(i) for i in tmp[1]]) + ']'

                    fcntl.flock(f, fcntl.LOCK_EX)
                    print(str_f, file=f)
                    fcntl.flock(f, fcntl.LOCK_UN)
    except:
        pass

def get_ipid(packet):
    global f,DICT,count
    try:
        src=packet['IPv6'].src
        if 'IPv6 Extension Header - Fragmentation header' in packet:
            ipid=packet['IPv6 Extension Header - Fragmentation header'].id


            str_f=src + ' ' + str(ipid)

            fcntl.flock(f, fcntl.LOCK_EX)
            print(str_f, file=f)
            fcntl.flock(f, fcntl.LOCK_UN)
    except:
        pass     

if __name__ == '__main__':
    f_name = ''               #result file
    f = open(f_name , 'a+', encoding='utf-8')
    rounds=15                 #rounds A and B send, we will get 2*rounds IP-ID per target ip
    source_A=''               #IPv6 address of host A
    source_B=''               #IPv6 address of host B

    DICT=collections.OrderedDict()

    sniff(filter='dst host %s or %s and icmp6'%(source_A,source_B),prn=IPID_Write,iface='wlo1',store=0)
    while DICT:
        tmp=DICT.popitem(last=False)
        str_f=tmp[0] +' [' +' '.join([str(i) for i in tmp[1]]) + ']'
        fcntl.flock(f, fcntl.LOCK_EX)
        print(str_f, file=f)
        fcntl.flock(f, fcntl.LOCK_UN)
    f.close()
