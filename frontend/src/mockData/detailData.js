export const mockDetailData = {
    // Token Info
    id: 1,
    address: "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
    tokenName: "BitDao",
    symbol: "BDO",
    tokenType: "ERC-20",
    contractOwner: "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
    pair: "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",

    // Risk Score 
    riskScore: 92,

    // Holder Info
    holders: [
        {rank: 1, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 85},
        {rank: 2, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 8},
        {rank: 3, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 1.5}
    ],

    // Scam Type
    scamTypes: [
        { type: "Honeypot", level: "Warning" },
        { type: "Exit", level: "Critical" }
    ],

    // Victim Insights
    victimInsights: [
        { 
            title: "코드 분석: Blacklist",
            description: "일반 사용자는 매도 불가한 blacklist 로직이 존재함",
        },
        {
            title: "코드 분석: 외부라우터호출",
            description: "외부 라우터를 호출하는 로직이 존재함"
        }
    ]
}

export default mockDetailData; 