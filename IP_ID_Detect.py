import multiprocessing
import random
import string
import time
import ipaddress

from scapy.all import *

def is_ipv6(address):
    try:
        ipaddress.IPv6Address(address)
        return True
    except:
        return False



def send_icmp_request(source,addr, data, seq=0):
    """Send echo request out

    Arguments:
        source {str} -- source address
        addr {str} -- target address
        data {str} -- payload

    Keyword Arguments:
        seq {int} -- sequence number in the ping request (default: {0})

    """

    if len(data)<1500 :
        base = IPv6(src=source,dst=addr, plen=len(data) + 8)
        extension = ICMPv6EchoRequest(data=data, seq=seq)
        packet = base / extension
        send(packet, verbose=False,iface='wlo1')
    else:
        base = IPv6(src=source,dst=addr, plen=len(data) + 8)
        extension = ICMPv6EchoRequest(data=data, seq=seq)
        packet = base / extension
        p=IPv6(raw(packet))
        checksum=p['ICMPv6 Echo Request'].cksum
    
        d1=data[:1224]
        d2=data[1224:]
        e1=ICMPv6EchoRequest(data=d1, seq=seq,cksum=checksum)
        f1=IPv6ExtHdrFragment(nh=58,offset=0,m=1,id=seq)
        b1=IPv6(nh=44,src=source,dst=addr, plen=len(d1) + 16)
        p1=b1/f1/e1
    
        e2=d2.encode("utf-8")
        f2=IPv6ExtHdrFragment(nh=58,offset=154,m=0,id=seq)
        b2=IPv6(nh=44,src=source,dst=addr, plen=len(d2) + 8)
        p2=b2/f2/e2
        send(p1, verbose=False,iface='wlo1')
        send(p2, verbose=False,iface='wlo1')



def send_too_big(source,addr, data, mtu=1280):
    """Send too big packet ICMPv6 packet.

    Arguments:
        source {str} -- source address
        addr {str} -- target address
        data {str} -- payload

    Keyword Arguments:
        seq {int} -- sequence number in the ping request (default: {0})
    """
    
    base = IPv6(src=addr,dst=source, plen=len(data) + 8,tc=4,hlim=48)
    extension = ICMPv6EchoReply(data=data, seq=0)
    packet = base / extension
    p=IPv6(raw(packet))
    checksum=p['ICMPv6 Echo Reply'].cksum

    too_big_extension = ICMPv6PacketTooBig(mtu=mtu) / \
        (base / ICMPv6EchoReply(data=data[:mtu - 96], seq=0,cksum=checksum))

    base = IPv6(src=source,dst=addr)

    too_big_packet = base / too_big_extension

    send(too_big_packet, verbose=False,iface='wlo1')




def random_generate_data(total_length):
    """Randomly generate data in length given.

    Arguments:
        total_length {int} -- length of the whole IPv6 Packet

    Returns:
        str -- data generated.
    """
    payload_length = total_length - 40
    data_length = payload_length - 8
    return ''.join(random.choices(string.ascii_letters + string.digits, k=data_length))



def solve_multiprocess(source_A,source_B,target_ips,rounds):
    """sending packets using multiprocessing

    Arguments:
        source_A {str} -- source address of host A
        source_B {str} -- source address of host B
        target_ips {list} -- list of ips
        rounds {int} -- rounds A and B send, we will get 2*rounds IP-ID per target ip
    """
    
    r=20
    data = random_generate_data(1300)
    length=len(target_ips)
    

    for j in range(0,length,r):
        round_ip=target_ips[j:j+r]
        l=len(round_ip)-1

        for m in range(4+2*rounds+l):
            for n in range(min(l,m)+1):
                if (m+n)%2 == 0:
                    sou=source_A
                else:
                    sou=source_B

                if m-n<=1 :
                    send_icmp_request(sou,(round_ip[n]).split()[0], data,seq=0)
                elif m-n<=3 :
                    send_too_big(sou,(round_ip[n]).split()[0], data, mtu=1000)
                elif m-n<=33 :
                    send_icmp_request(sou,(round_ip[n]).split()[0], data,seq=m-n-3)
                else:
                    pass
    



def run(targetfile,source_A,source_B,process_number=50,rounds=15):
    with open(targetfile, 'r', encoding='utf-8') as input_stream:
        p = multiprocessing.Pool(process_number)
        lines = input_stream.readlines()
        random.shuffle(lines)
        batch_size=len(lines)
        m=time.time()

        for i in range(process_number):
            t=batch_size//process_number
            if i == process_number-1:
                target_ips = lines[i*t:]
            else:
                target_ips = lines[i*t:i*t+t]
            
            p.apply_async(solve_multiprocess, args=(source_A,source_B,target_ips,rounds))
        p.close()
        p.join()
        n=time.time()
        print(n-m)


if __name__ == '__main__':
    targetfile=''    #file contains target ip, each line is a IPv6 address 
    source_A=''      #IPv6 address of host A
    source_B=''      #IPv6 address of host B
    run(targetfile,source_A,source_B,50,15)






