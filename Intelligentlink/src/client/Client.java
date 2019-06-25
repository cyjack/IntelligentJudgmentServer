package client;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;

import thrift_api.IntelligentJudgment;

public class Client {
	TTransport transport;
	TProtocol protocol;
	IntelligentJudgment.Client client;
	public static void main(String[] args) {
		// TODO Auto-generated method stub
//		String[] answer = {"因为小鸟不喜欢吃肉喝酒。",
//			               "鲁王喝醉了",
//			               "因为他没有按照小鸟的生活安排。",
//				           "人们有自己的规律，而鸟也有鸟的规律。",
//				           "因为鲁国国王没有符合自然的规律，鸟类应该让它栖息在深山老林，在陆地沙洲游玩，在江河湖海飞翔。"};
//		String q_id = "2017_01_qm_luwang";
//		Client client = new Client();
//		for(int i=0; i<answer.length; i++)
//			client.request(q_id, answer[i]);
		for(int z=1;z<100000000;z++){
			String q_id = "math_002";
			Client client = new Client();
			String answer = "老师可以在155~165的身高范围内挑选队员.因为在此范围内，人数最为集中，且大家的身高相对接近.";
			client.request(q_id, answer);
		}
		
	}
	/**
	 * Get score of answer content "q_content" for question "q_id"
	 * @param q_id ID of question, as "2016_08_qm_11_2", 
	 * 				where 2016 means year, 08 student grade,
	 * 				qm paper ID, 11_2 question number
	 * @param q_content answer content of question, as "人们有自己的规律，而鸟也有鸟的规律。"
	 * @return
	 * String
	 */
	public String request(String q_id, String q_content)
	{
		String result = "";
        try {
            transport = new TSocket("172.18.136.139", 8989);
            protocol = new TBinaryProtocol(transport);
            client = new IntelligentJudgment.Client(protocol);
            transport.open(); 
            result = client.judgment(q_id, q_content);
            System.out.println(q_id+" answer: "+q_content);
            System.out.println(q_id+" score: "+result);
            transport.close();
        } catch (TTransportException e) {
            e.printStackTrace();
        } catch (TException e) {
            e.printStackTrace();
        }
        return result;
	}
}
