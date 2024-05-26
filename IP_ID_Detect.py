import multiprocessing
import random
import re
import string
import tqdm
import time

from scapy.all import *


def send_echo(source,addr, data, seq=0):        #发送ICMPv6 echo Request包，超过1500需要分片发送

    if len(data)<1500 :
        base = IPv6(src=source,dst=addr, plen=len(data) + 8)
        extension = ICMPv6EchoRequest(data=data, seq=seq)
        packet = base / extension
        send(packet, verbose=False,iface='ppp0')
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
        send(p1, verbose=False,iface='ppp0')
        send(p2, verbose=False,iface='ppp0')



def send_too_big(source,addr, data, mtu=1280):     #发送ICMPv6 Too Big包，超过1500需要分片发送
    
    base = IPv6(src=addr,dst=source, plen=len(data) + 8,tc=4,hlim=48)
    extension = ICMPv6EchoReply(data=data, seq=0)
    packet = base / extension
    p=IPv6(raw(packet))
    checksum=p['ICMPv6 Echo Reply'].cksum

    too_big_extension = ICMPv6PacketTooBig(mtu=mtu) / \
        (base / ICMPv6EchoReply(data=data[:mtu - 96], seq=0,cksum=checksum))

    base = IPv6(src=source,dst=addr)

    too_big_packet = base / too_big_extension

    send(too_big_packet, verbose=False,iface='ppp0')




def random_generate_data(total_length):       #填充数据包内容
    
    payload_length = total_length - 40
    data_length = payload_length - 8
    return ''.join(random.choices(string.ascii_letters + string.digits, k=data_length))



def solve_multiprocess(source_A,source_B,target_ips,rounds):
    
    data = random_generate_data(1300)
 
    length=len(target_ips)
    
    for j in range(length):
        target_ip=target_ips[j][:-1]      
        send_echo(source_A,target_ip, data)
        send_echo(source_B,target_ip, data) 

        send_too_big(source_A,target_ip, data, mtu=1280)
        send_too_big(source_B,target_ip, data, mtu=1280)
    
        
        for i in range(rounds):
            send_echo(source_A,target_ip, data, seq=2*i+1)
            send_echo(source_B,target_ip, data, seq=2*i+2)




def run(source_A,source_B,targetfile,process_number=50,batch_size=20000,rounds=15):
    with open(targetfile, 'r', encoding='utf-8') as input_stream:
        p = multiprocessing.Pool(process_number)
        lines = input_stream.readlines()
        random.shuffle(lines)
        m=time.time()

        for i in range(process_number):
            t=batch_size//process_number
            if i == process_number-1:
                target_ips = lines[i*t:]
            else:
                target_ips = lines[i*t:i*t+t]
            
            p.apply_async(solve_multiprocess, args=(
                target_ips,rounds,source_A,source_B,))
        p.close()
        p.join()
        n=time.time()
        print(n-m)


if __name__ == '__main__':
    source_A=''
    source_B=''
    targetfile=''
    run(source_A,source_B,targetfile,process_number=50, batch_size=1000000,rounds=15)


