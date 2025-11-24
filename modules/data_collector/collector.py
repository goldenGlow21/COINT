"""
Unified data collector for Ethereum token analysis.
Collects token info, pair events, and holder information.
"""
import logging
from typing import Dict, List, Any
from decimal import Decimal
from datetime import datetime

from web3 import Web3
from web3.exceptions import ContractLogicError
import requests

logger = logging.getLogger(__name__)


# ERC20 ABI (minimal)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]

# Uniswap V2 Factory ABI (minimal)
UNISWAP_V2_FACTORY_ABI = [
    {
        "constant": True,
        "inputs": [
            {"name": "tokenA", "type": "address"},
            {"name": "tokenB", "type": "address"}
        ],
        "name": "getPair",
        "outputs": [{"name": "pair", "type": "address"}],
        "type": "function"
    }
]

# Uniswap V2 Pair ABI (minimal + events)
UNISWAP_V2_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "reserve0", "type": "uint112"},
            {"name": "reserve1", "type": "uint112"},
            {"name": "blockTimestampLast", "type": "uint32"}
        ],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": False, "name": "amount0", "type": "uint256"},
            {"indexed": False, "name": "amount1", "type": "uint256"}
        ],
        "name": "Mint",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": False, "name": "amount0", "type": "uint256"},
            {"indexed": False, "name": "amount1", "type": "uint256"},
            {"indexed": True, "name": "to", "type": "address"}
        ],
        "name": "Burn",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": False, "name": "amount0In", "type": "uint256"},
            {"indexed": False, "name": "amount1In", "type": "uint256"},
            {"indexed": False, "name": "amount0Out", "type": "uint256"},
            {"indexed": False, "name": "amount1Out", "type": "uint256"},
            {"indexed": True, "name": "to", "type": "address"}
        ],
        "name": "Swap",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "name": "reserve0", "type": "uint112"},
            {"indexed": False, "name": "reserve1", "type": "uint112"}
        ],
        "name": "Sync",
        "type": "event"
    }
]

# Uniswap V2 Factory address on Ethereum mainnet
UNISWAP_V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
# WETH address on Ethereum mainnet
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"


class UnifiedDataCollector:
    """Unified collector for token info, pair events, and holders."""

    def __init__(self, rpc_url: str, etherscan_api_key: str, etherscan_api_url: str):
        """
        Initialize the data collector.

        Args:
            rpc_url: Ethereum RPC endpoint URL
            etherscan_api_key: Etherscan API key
            etherscan_api_url: Etherscan API base URL
        """
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.etherscan_api_key = etherscan_api_key
        self.etherscan_api_url = etherscan_api_url

        if not self.web3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")

        logger.info(f"Connected to Ethereum node (chain_id={self.web3.eth.chain_id})")

    def collect_all(self, token_addr: str) -> Dict[str, Any]:
        """
        Collect all data for a given token address.

        Args:
            token_addr: Token contract address (checksummed)

        Returns:
            Dictionary containing:
                - token_info: Dict with token metadata
                - pair_events: List of pair event logs
                - holders: List of holder addresses and balances
        """
        token_addr = Web3.to_checksum_address(token_addr)
        logger.info(f"Collecting data for token: {token_addr}")

        # 1. Collect token info
        token_info = self._collect_token_info(token_addr)

        # 2. Find pair address and collect events
        pair_addr = self._find_pair_address(token_addr)
        pair_events = []
        if pair_addr and pair_addr != "0x0000000000000000000000000000000000000000":
            pair_events = self._collect_pair_events(pair_addr)
        else:
            logger.warning(f"No Uniswap V2 pair found for token {token_addr}")

        # 3. Collect holders
        holders = self._collect_holders(token_addr)

        return {
            'token_info': token_info,
            'pair_events': pair_events,
            'holders': holders
        }

    def _collect_token_info(self, token_addr: str) -> Dict[str, Any]:
        """Collect token metadata from ERC20 contract."""
        logger.info(f"Collecting token info for {token_addr}")

        contract = self.web3.eth.contract(address=token_addr, abi=ERC20_ABI)

        try:
            name = contract.functions.name().call()
        except (ContractLogicError, Exception) as e:
            logger.warning(f"Failed to get token name: {e}")
            name = ""

        try:
            symbol = contract.functions.symbol().call()
        except (ContractLogicError, Exception) as e:
            logger.warning(f"Failed to get token symbol: {e}")
            symbol = ""

        try:
            decimals = contract.functions.decimals().call()
        except (ContractLogicError, Exception) as e:
            logger.warning(f"Failed to get token decimals: {e}")
            decimals = 18

        try:
            total_supply = contract.functions.totalSupply().call()
        except (ContractLogicError, Exception) as e:
            logger.warning(f"Failed to get token totalSupply: {e}")
            total_supply = 0

        # Get contract creator from Etherscan
        creator_addr = self._get_contract_creator(token_addr)

        return {
            'token_addr': token_addr,
            'token_name': name,
            'token_symbol': symbol,
            'token_decimals': decimals,
            'token_total_supply': str(total_supply),
            'token_creator_addr': creator_addr,
            'created_at': datetime.now()
        }

    def _get_contract_creator(self, contract_addr: str) -> str:
        """Get contract creator address from Etherscan API."""
        logger.info(f"Fetching contract creator for {contract_addr}")

        params = {
            'module': 'contract',
            'action': 'getcontractcreation',
            'contractaddresses': contract_addr,
            'apikey': self.etherscan_api_key
        }

        try:
            response = requests.get(self.etherscan_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == '1' and data.get('result'):
                creator = data['result'][0].get('contractCreator', '')
                logger.info(f"Contract creator: {creator}")
                return creator
            else:
                logger.warning(f"Failed to get contract creator: {data.get('message', 'Unknown error')}")
                return ""
        except Exception as e:
            logger.error(f"Error fetching contract creator: {e}")
            return ""

    def _find_pair_address(self, token_addr: str) -> str:
        """Find Uniswap V2 pair address for token/WETH."""
        logger.info(f"Finding Uniswap V2 pair for {token_addr}")

        factory = self.web3.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_V2_FACTORY),
            abi=UNISWAP_V2_FACTORY_ABI
        )

        try:
            pair_addr = factory.functions.getPair(token_addr, Web3.to_checksum_address(WETH_ADDRESS)).call()
            logger.info(f"Found pair address: {pair_addr}")
            return pair_addr
        except Exception as e:
            logger.error(f"Error finding pair address: {e}")
            return "0x0000000000000000000000000000000000000000"

    def _collect_pair_events(self, pair_addr: str) -> List[Dict[str, Any]]:
        """Collect Mint, Burn, Swap, Sync events from Uniswap V2 pair."""
        logger.info(f"Collecting pair events for {pair_addr}")

        pair_contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(pair_addr),
            abi=UNISWAP_V2_PAIR_ABI
        )

        # Get current block
        current_block = self.web3.eth.block_number
        # Collect events from the last 10000 blocks (adjust as needed)
        from_block = max(0, current_block - 10000)

        events = []
        event_names = ['Mint', 'Burn', 'Swap', 'Sync']

        for event_name in event_names:
            try:
                event_filter = getattr(pair_contract.events, event_name).create_filter(
                    fromBlock=from_block,
                    toBlock='latest'
                )
                logs = event_filter.get_all_entries()

                for log in logs:
                    # Get LP total supply at the event block
                    lp_total_supply = pair_contract.functions.totalSupply().call(
                        block_identifier=log['blockNumber']
                    )

                    events.append({
                        'pair_addr': pair_addr,
                        'evt_name': event_name,
                        'evt_log': {
                            'transactionHash': log['transactionHash'].hex(),
                            'blockNumber': log['blockNumber'],
                            'blockHash': log['blockHash'].hex(),
                            'logIndex': log['logIndex'],
                            'args': {k: str(v) for k, v in log['args'].items()}
                        },
                        'lp_total_supply': str(lp_total_supply),
                        'created_at': datetime.now()
                    })

                logger.info(f"Collected {len(logs)} {event_name} events")
            except Exception as e:
                logger.error(f"Error collecting {event_name} events: {e}")

        return events

    def _collect_holders(self, token_addr: str, page: int = 1, offset: int = 100) -> List[Dict[str, Any]]:
        """Collect token holders from Etherscan API."""
        logger.info(f"Collecting holders for {token_addr}")

        params = {
            'module': 'token',
            'action': 'tokenholderlist',
            'contractaddress': token_addr,
            'page': page,
            'offset': offset,
            'apikey': self.etherscan_api_key
        }

        try:
            response = requests.get(self.etherscan_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == '1' and data.get('result'):
                holders = []
                for holder in data['result']:
                    holders.append({
                        'holder_addr': holder.get('TokenHolderAddress', ''),
                        'balance': holder.get('TokenHolderQuantity', '0'),
                        'created_at': datetime.now()
                    })

                logger.info(f"Collected {len(holders)} holders")
                return holders
            else:
                logger.warning(f"Failed to get holders: {data.get('message', 'Unknown error')}")
                return []
        except Exception as e:
            logger.error(f"Error fetching holders: {e}")
            return []
