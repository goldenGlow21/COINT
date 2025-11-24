#!/usr/bin/python3
"""
Trading Suspend Detection Scenario
컨트랙트의 거래 중단(Trade Blocking) 기능 동적 탐지
"""

from brownie import accounts, chain, Contract, network
# 공통 상수
GAS_PRICE = "100 gwei"
BUY_AMOUNT = int(0.1 * 1e18)
ETH_TRANSFER_AMOUNT = int(1.0 * 1e18)
SELL_RATIO = 0.5
GAS_LIMIT = ""
def prepare_account_with_quote(detector, account, eth_amount_for_quote):
    """계정에 Quote Token 준비 (basic_swap_test 방식)

    Args:
        eth_amount_for_quote: Quote token 확보를 위해 사용할 ETH 양 (wei)
    """
    quote_addr = detector.quote_token_address or detector.weth.address

    # 1. ETH 잔고 확보 (가스비 포함)
    total_eth_needed = eth_amount_for_quote + int(5.0 * 1e18)  # 5 ETH 가스비 여유 (높은 gas_limit 대비)
    if account.balance() < total_eth_needed:
        accounts[0].transfer(account, total_eth_needed - account.balance(), gas_price=GAS_PRICE)
        network.web3.provider.make_request("anvil_mine", [1])

    # 2. WETH deposit
    detector.weth.deposit({"from": account, "value": eth_amount_for_quote, "gas_price": GAS_PRICE, "gas_limit":GAS_LIMIT})
    network.web3.provider.make_request("anvil_mine", [1])

    # 3. WETH인 경우 완료
    if quote_addr.lower() == detector.weth.address.lower():
        return detector.weth, quote_addr

    # 4. WETH가 아닌 경우: WETH 전체를 Quote Token으로 swap
    deadline = chain.time() + 300
    weth_balance = detector.weth.balanceOf(account.address)

    detector.weth.approve(detector.router_addr, weth_balance, {"from": account, "gas_price": GAS_PRICE, "gas_limit":GAS_LIMIT})
    network.web3.provider.make_request("anvil_mine", [1])

    detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
        weth_balance,
        0,
        [detector.weth.address, detector.quote_token_address],
        account.address,
        deadline,
        {"from": account, "gas_price": GAS_PRICE, "gas_limit":GAS_LIMIT}
    )
    network.web3.provider.make_request("anvil_mine", [1])

    return detector.quote_token, detector.quote_token_address

def get_abi_from_etherscan(token_address):
    """Deprecated stub kept for backward compatibility."""
    return None

def has_function_in_abi(abi, function_name):
    """ABI에 특정 함수가 존재하는지 확인"""
    if not abi:
        return False
    return any(item.get("type") == "function" and item.get("name") == function_name for item in abi)

def find_suspension_functions(abi):
    """거래 중단 관련 함수들을 ABI에서 찾음"""
    suspension_functions = {
        "pause": [
            "pause", "setPaused", "pauseTrading", "pauseContract",
            "pauseToken", "setPause", "setContractPaused"," "
        ],
        "unpause": [
            "unpause", "resumeTrading", "unpauseTrading", "unpauseContract",
            "unpauseToken", "setUnpaused"
        ],
        "enableTrading": [
            "enableTrading", "setTradingEnabled", "openTrading", "tradeStatus",
            "setTradeStatus", "setTradeEnabled", "activateTrading", "setTradingActive",
            "toggleTrading", "switchTrading", "startTrading", "launchTrading",
            "setTradable", "enableTrade", "setCanTrade", "setTradingOpen",
            # 일반적 이름 (거래 차단 악용 가능)
            "Execute", "execute", "Multicall", "multicall", "Batch", "batch",
            # Cooldown 위장 (실제로는 거래 차단)
            "useCoolDown", "useCooldown", "setCoolDown", "setCooldown"
        ],
        "disableTrading": [
            "disableTrading", "stopTrading", "setTradingDisabled", "closeTrading",
            "deactivateTrading", "haltTrading", "freezeTrading", "lockTrading",
            "setTradingClosed"
        ]
    }

    found = {}
    for category, func_names in suspension_functions.items():
        for func_name in func_names:
            if has_function_in_abi(abi, func_name):
                found[category] = func_name
                break

    return found

def run_scenario(detector):
    global GAS_LIMIT
    """Trading Suspend 탐지 시나리오 실행"""
    result = {
        "scenario": "trading_suspend",
        "result": "NO",
        "confidence": "LOW",
        "details": {
            "owner_address": None,
            "owner_renounced": False,
            "suspension_functions_found": {},
            "buyer1_initial_buy": False,
            "suspension_executed": False,
            "buyer1_sell_after_suspend": None,
            "buyer2_buy_after_suspend": None,
            "buyer2_sell_after_suspend": None,
            "owner_trade_after_suspend": None,
            "evidence": []
        },
        "reason": ""
    }
    GAS_LIMIT = detector.gaslimit
    network.gas_limit(detector.gaslimit)
    
    try:
        print(f"\n{'='*60}")
        print("Trading Suspend Detection Scenario")
        print(f"{'='*60}")

        # Phase 1: Owner 확인
        print("\n[Phase 1] Owner 확인")
        # owner_address = detector.token_owner or detector.find_owner()
        print(detector.owner_candidate)
        for owner in detector.owner_candidate:
            if owner is not None:
                owner_address = owner
                break
            else:
                continue
        
        result["details"]["owner_address"] = owner_address

        if not owner_address:
            result["result"] = "NO"
            result["confidence"] = "MEDIUM"
            result["reason"] = "Owner 확인 불가 - 거래 중단 가능성 낮음"
            print(f"  Owner 확인 불가")
            return result

        if owner_address == "0x0000000000000000000000000000000000000000":
            result["details"]["owner_renounced"] = True
            result["result"] = "NO"
            result["confidence"] = "HIGH"
            result["reason"] = "Owner가 renounced됨 - 거래 중단 권한 없음"
            print(f"  Owner renounced: {owner_address}")
            return result

        owner_account = accounts.at(owner_address, force=True)
        print(f"  ✅ Owner 발견: {owner_address}")

        privileged_accounts = [owner_account]
        seen_accounts = {owner_address.lower()}
        if hasattr(detector, "owner_candidate"):
            for candidate in detector.owner_candidate:
                if not candidate:
                    continue
                candidate_lower = candidate.lower()
                if candidate_lower in seen_accounts:
                    continue
                try:
                    candidate_account = accounts.at(candidate, force=True)
                    privileged_accounts.append(candidate_account)
                    seen_accounts.add(candidate_lower)
                    print(f"  추가 권한 후보 확보: {candidate}")
                except Exception as candidate_error:
                    print(f"  추가 후보 impersonate 실패 ({candidate}): {type(candidate_error).__name__}")

        # Phase 2: 거래 중단 함수 탐지
        print("\n[Phase 2] 거래 중단 함수 탐지")
        abi = getattr(detector.token, "abi", None)
        if not abi:
            print("  Token ABI unavailable, falling back to default interface")
            abi = detector.token_abi

        suspension_funcs = find_suspension_functions(abi)
        result["details"]["suspension_functions_found"] = suspension_funcs

        if suspension_funcs:
            print(f"  ✅ 거래 중단 함수 발견:")
            for category, func_name in suspension_funcs.items():
                print(f"    - {category}: {func_name}()")
                result["details"]["evidence"].append(f"{category}:{func_name}")
        else:
            print("  거래 중단 함수를 찾지 못함")
            result["result"] = "NO"
            result["confidence"] = "MEDIUM"
            result["reason"] = "거래 중단 함수 없음 - 동적 테스트 불가"
            return result

        token_with_full_abi = Contract.from_abi("Token", detector.token_address, abi)

        def ensure_account_funded(acct):
            # 높은 gas_limit를 고려하여 충분한 ETH 확보 (최소 5 ETH)
            min_balance = int(5.0 * 1e18)
            if acct.balance() < min_balance:
                try:
                    accounts[0].transfer(acct.address, min_balance - acct.balance(), gas_price=GAS_PRICE,gas_limit=GAS_LIMIT)
                except Exception as funding_error:
                    print(f"  {acct.address} 가스 충전 실패: {type(funding_error).__name__}")

        def execute_with_privileged(callable_fn):
            last_error = None
            for acct in privileged_accounts:
                ensure_account_funded(acct)
                try:
                    callable_fn(acct)
                    return acct
                except Exception as call_error:
                    print(f"     {acct.address[:10]}... 시도 실패: {type(call_error).__name__}")
                    last_error = call_error
            if last_error:
                raise last_error
            raise Exception("No privileged account available")

        # Phase 3: Buyer1 초기 매수 테스트
        print("\n[Phase 3] Buyer1 초기 매수 테스트")
        buyer1 = accounts[1]
        print(f"  Buyer1: {buyer1.address}")

        # Quote Token 준비
        quote_token_buyer1, quote_addr = prepare_account_with_quote(detector, buyer1, ETH_TRANSFER_AMOUNT)
        quote_balance_buyer1 = quote_token_buyer1.balanceOf(buyer1.address)
        path_reverse = [detector.token_address, quote_addr]

        try:
            deadline = chain.time() + 300
            path = [quote_addr, detector.token_address]

            # 받은 quote token의 10% 사용 (decimals 무관)
            buy_amount = quote_balance_buyer1 // 10

            quote_token_buyer1.approve(detector.router_addr, buy_amount, {"from": buyer1, "gas_price": GAS_PRICE})
            network.web3.provider.make_request("anvil_mine", [1])

            detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                buy_amount, 0, path, buyer1.address, deadline,
                {"from": buyer1, "gas_price": GAS_PRICE}
            )
            network.web3.provider.make_request("anvil_mine", [1])

            balance = detector.token.balanceOf(buyer1.address)
            if balance > 0:
                print(f"  ✅ Buyer1 매수 성공 (잔액: {balance / 1e18:.4f})")
                result["details"]["buyer1_initial_buy"] = True
            else:
                print(f"  ⚠️  Buyer1 매수 실패 (잔액 0)")
        except Exception as e:
            print(f"  ⚠️  Buyer1 매수 실패: {type(e).__name__}")
            result["details"]["evidence"].append("buyer1_initial_blocked")

        # Phase 4: Owner가 거래 통제 함수 호출
        print("\n[Phase 4] Owner가 거래 통제 함수 호출")
        suspension_executed = False
        executed_function = None

        # pause 계열 함수 시도
        if not suspension_executed and "pause" in suspension_funcs:
            func_name = suspension_funcs["pause"]
            try:
                func = getattr(token_with_full_abi, func_name)
                func_abi = next((item for item in abi if item.get("name") == func_name), None)

                def call_with_account(acct):
                    if func_abi and len(func_abi.get("inputs", [])) == 0:
                        func({"from": acct, "gas_price": GAS_PRICE})
                    elif func_abi and len(func_abi.get("inputs", [])) == 1:
                        func(True, {"from": acct, "gas_price": GAS_PRICE})
                    else:
                        raise Exception("Unknown function signature")

                executor = execute_with_privileged(call_with_account)
                print(f"  ✅ {func_name}() 실행 성공 (caller: {executor.address})")
                suspension_executed = True
                executed_function = func_name
                result["details"]["suspension_executed"] = True
                result["details"]["evidence"].append(f"executed:{func_name}:{executor.address}")
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                print(f"  ⚠️  {func_name}() 실행 실패: {type(e).__name__}")
                print(f"     에러 상세: {error_msg}")

        # disableTrading 계열 함수 시도
        if not suspension_executed and "disableTrading" in suspension_funcs:
            func_name = suspension_funcs["disableTrading"]
            try:
                func = getattr(token_with_full_abi, func_name)
                func_abi = next((item for item in abi if item.get("name") == func_name), None)

                def call_with_account(acct):
                    if func_abi and len(func_abi.get("inputs", [])) == 0:
                        func({"from": acct, "gas_price": GAS_PRICE})
                    elif func_abi and len(func_abi.get("inputs", [])) == 1:
                        func(False, {"from": acct, "gas_price": GAS_PRICE})
                    else:
                        raise Exception("Unknown function signature")

                executor = execute_with_privileged(call_with_account)
                print(f"  ✅ {func_name}() 실행 성공 (caller: {executor.address})")
                suspension_executed = True
                executed_function = func_name
                result["details"]["suspension_executed"] = True
                result["details"]["evidence"].append(f"executed:{func_name}:{executor.address}")
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                print(f"  ⚠️  {func_name}() 실행 실패: {type(e).__name__}")
                print(f"     에러 상세: {error_msg}")

        # enableTrading 계열 함수 시도 (true/false 둘 다)
        if not suspension_executed and "enableTrading" in suspension_funcs:
            func_name = suspension_funcs["enableTrading"]
            try:
                func = getattr(token_with_full_abi, func_name)
                func_abi = next((item for item in abi if item.get("name") == func_name), None)

                if func_abi and len(func_abi.get("inputs", [])) == 1:
                    try:
                        executor = execute_with_privileged(lambda acct: func(False, {"from": acct, "gas_price": GAS_PRICE}))
                        print(f"  ✅ {func_name}(false) 실행 성공 (caller: {executor.address})")
                        suspension_executed = True
                        executed_function = func_name
                        result["details"]["suspension_executed"] = True
                        result["details"]["evidence"].append(f"executed:{func_name}(false):{executor.address}")
                    except Exception:
                        executor = execute_with_privileged(lambda acct: func(True, {"from": acct, "gas_price": GAS_PRICE}))
                        print(f"  ✅ {func_name}(true) 실행 성공 (caller: {executor.address})")
                        suspension_executed = True
                        executed_function = func_name
                        result["details"]["suspension_executed"] = True
                        result["details"]["evidence"].append(f"executed:{func_name}(true):{executor.address}")
                elif func_abi and len(func_abi.get("inputs", [])) == 0:
                    executor = execute_with_privileged(lambda acct: func({"from": acct, "gas_price": GAS_PRICE}))
                    print(f"  ✅ {func_name}() 실행 성공 (caller: {executor.address})")
                    suspension_executed = True
                    executed_function = func_name
                    result["details"]["suspension_executed"] = True
                    result["details"]["evidence"].append(f"executed:{func_name}:{executor.address}")
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                print(f"  ⚠️  {func_name}() 실행 실패: {type(e).__name__}")
                print(f"     에러 상세: {error_msg}")

        if not suspension_executed:
            print("  ⚠️  모든 거래 통제 함수 실행 실패")

        # Phase 5: Buyer1 매도 시도
        print("\n[Phase 5] Buyer1 매도 시도")
        buyer1_balance = detector.token.balanceOf(buyer1.address)

        if buyer1_balance > 0:
            try:
                sell_amount = int(buyer1_balance * SELL_RATIO)
                detector.token.approve(detector.router_addr, sell_amount, {"from": buyer1, "gas_price": GAS_PRICE})

                deadline = chain.time() + 300
                detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                    sell_amount, 0, path_reverse, buyer1.address, deadline,
                    {"from": buyer1, "gas_price": GAS_PRICE}
                )

                print(f"  ✅ Buyer1 매도 성공")
                result["details"]["buyer1_sell_after_suspend"] = True
            except Exception as e:
                print(f"  ❌ Buyer1 매도 실패: {type(e).__name__}")
                result["details"]["buyer1_sell_after_suspend"] = False
                result["details"]["evidence"].append("buyer1_sell_blocked")
        else:
            print(f"  ⚠️  Buyer1 잔액 없음 - 매도 스킵")
            result["details"]["buyer1_sell_after_suspend"] = None

        # Phase 6: Buyer2 매수/매도 시도 (중단 후)
        print("\n[Phase 6] Buyer2 매수/매도 시도 (거래 중단 후)")
        buyer2 = accounts[2]
        print(f"  Buyer2: {buyer2.address}")

        # Quote Token 준비
        quote_token_buyer2, _ = prepare_account_with_quote(detector, buyer2, ETH_TRANSFER_AMOUNT)
        quote_balance_buyer2 = quote_token_buyer2.balanceOf(buyer2.address)
        path_reverse_buyer2 = [detector.token_address, quote_addr]

        try:
            deadline = chain.time() + 300
            path = [quote_addr, detector.token_address]

            # 받은 quote token의 10% 사용 (decimals 무관)
            buy_amount_buyer2 = quote_balance_buyer2 // 10

            quote_token_buyer2.approve(detector.router_addr, buy_amount_buyer2, {"from": buyer2, "gas_price": GAS_PRICE})
            network.web3.provider.make_request("anvil_mine", [1])

            detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                buy_amount_buyer2, 0, path, buyer2.address, deadline,
                {"from": buyer2, "gas_price": GAS_PRICE}
            )
            network.web3.provider.make_request("anvil_mine", [1])

            buyer2_balance = detector.token.balanceOf(buyer2.address)
            if buyer2_balance > 0:
                print(f"  ✅ Buyer2 매수 성공 (잔액: {buyer2_balance / 1e18:.4f})")
                result["details"]["buyer2_buy_after_suspend"] = True

                try:
                    sell_amount2 = int(buyer2_balance * SELL_RATIO)
                    detector.token.approve(detector.router_addr, sell_amount2, {"from": buyer2, "gas_price": GAS_PRICE})

                    detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                        sell_amount2, 0, path_reverse_buyer2, buyer2.address, deadline,
                        {"from": buyer2, "gas_price": GAS_PRICE}
                    )

                    print(f"  ✅ Buyer2 매도 성공")
                    result["details"]["buyer2_sell_after_suspend"] = True
                except Exception as e2:
                    print(f"  ❌ Buyer2 매도 실패: {type(e2).__name__}")
                    result["details"]["buyer2_sell_after_suspend"] = False
                    result["details"]["evidence"].append("buyer2_sell_blocked")
            else:
                print(f"  ❌ Buyer2 매수 실패 (잔액 0)")
                result["details"]["buyer2_buy_after_suspend"] = False
                result["details"]["evidence"].append("buyer2_blocked")
        except Exception as e:
            print(f"  ❌ Buyer2 매수 실패: {type(e).__name__}")
            result["details"]["buyer2_buy_after_suspend"] = False
            result["details"]["evidence"].append("buyer2_blocked")

        # Phase 7: Owner 거래 가능 여부 확인
        print("\n[Phase 7] Owner 거래 가능 여부 확인")

        # Quote token 준비
        quote_token, quote_addr = prepare_account_with_quote(detector, owner_account, ETH_TRANSFER_AMOUNT)

        try:
            deadline = chain.time() + 300
            path = [quote_addr, detector.token_address]

            quote_balance = quote_token.balanceOf(owner_account.address)
            # 받은 quote token의 10% 사용 (decimals 무관)
            quote_amount = quote_balance // 10

            quote_token.approve(detector.router_addr, quote_amount, {"from": owner_account, "gas_price": GAS_PRICE})
            network.web3.provider.make_request("anvil_mine", [1])

            detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                quote_amount, 0, path, owner_account.address, deadline,
                {"from": owner_account, "gas_price": GAS_PRICE}
            )
            network.web3.provider.make_request("anvil_mine", [1])

            owner_balance = detector.token.balanceOf(owner_account.address)
            if owner_balance > 0:
                print(f"  ✅ Owner 매수 성공 (잔액: {owner_balance / 1e18:.4f})")

                try:
                    sell_amount_owner = int(owner_balance * SELL_RATIO)
                    detector.token.approve(detector.router_addr, sell_amount_owner, {"from": owner_account, "gas_price": GAS_PRICE})
                    network.web3.provider.make_request("anvil_mine", [1])

                    path_reverse = [detector.token_address, quote_addr]
                    detector.router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                        sell_amount_owner, 0, path_reverse, owner_account.address, deadline,
                        {"from": owner_account, "gas_price": GAS_PRICE}
                    )
                    network.web3.provider.make_request("anvil_mine", [1])

                    print(f"  ✅ Owner 매도 성공")
                    result["details"]["owner_trade_after_suspend"] = True
                    result["details"]["evidence"].append("owner_can_trade")
                except Exception as e2:
                    print(f"  ❌ Owner 매도 실패: {type(e2).__name__}")
                    result["details"]["owner_trade_after_suspend"] = False
            else:
                print(f"  ❌ Owner 매수 실패 (잔액 0)")
                result["details"]["owner_trade_after_suspend"] = False
        except Exception as e:
            print(f"  ❌ Owner 매수 실패: {type(e).__name__}")
            result["details"]["owner_trade_after_suspend"] = False

        # Phase 8: 최종 판정
        print("\n[Phase 8] 최종 판정")

        buyer1_initial_blocked = result["details"]["buyer1_initial_buy"] == False
        buyer1_sell_blocked = result["details"]["buyer1_sell_after_suspend"] == False
        buyer2_buy_blocked = result["details"]["buyer2_buy_after_suspend"] == False
        buyer2_sell_blocked = result["details"]["buyer2_sell_after_suspend"] == False
        owner_can_trade = result["details"]["owner_trade_after_suspend"] == True

        evidence_count = len(result["details"]["evidence"])

        print(f"  수집된 증거:")
        print(f"    - 거래 통제 함수 발견: {len(suspension_funcs)}개")
        print(f"    - 함수 실행 성공: {suspension_executed}")
        print(f"    - Buyer1 초기 차단: {buyer1_initial_blocked}")
        print(f"    - Buyer1 매도 차단: {buyer1_sell_blocked}")
        print(f"    - Buyer2 매수 차단: {buyer2_buy_blocked}")
        print(f"    - Buyer2 매도 차단: {buyer2_sell_blocked}")
        print(f"    - Owner 거래 가능: {owner_can_trade}")
        print(f"    - 총 증거: {evidence_count}개")

        users_blocked = buyer1_initial_blocked or buyer1_sell_blocked or buyer2_buy_blocked or buyer2_sell_blocked

        # 판정 로직
        if suspension_funcs and users_blocked and owner_can_trade:
            result["result"] = "YES"
            result["confidence"] = "HIGH"
            func_list = ", ".join([f"{k}:{v}" for k, v in suspension_funcs.items()])
            result["reason"] = f"거래 통제 함수 보유({func_list}) + Owner만 거래 가능 + 일반 사용자 차단 - 악의적 거래 통제"
        elif suspension_executed and users_blocked:
            result["result"] = "YES"
            result["confidence"] = "HIGH"
            result["reason"] = f"{executed_function}() 실행 + 일반 사용자 거래 차단 확인 - 거래 중단 기능 존재"
        elif suspension_funcs and users_blocked:
            result["result"] = "YES"
            result["confidence"] = "MEDIUM"
            func_list = ", ".join([f"{k}:{v}" for k, v in suspension_funcs.items()])
            result["reason"] = f"거래 통제 함수 보유({func_list}) + 일반 사용자 거래 차단 확인 - 거래 통제 가능"
        elif suspension_executed:
            result["result"] = "YES"
            result["confidence"] = "MEDIUM"
            result["reason"] = f"{executed_function}() 실행 성공 - 거래 통제 가능"
        elif suspension_funcs:
            # 함수는 있지만 실행 실패 & 차단 증거 없음 → 작동하지 않음
            result["result"] = "NO"
            result["confidence"] = "MEDIUM"
            func_list = ", ".join([f"{k}:{v}" for k, v in suspension_funcs.items()])
            result["reason"] = f"거래 통제 함수 발견({func_list})했으나 실행 실패 및 차단 증거 없음 - 기능 미작동"
        else:
            result["result"] = "NO"
            result["confidence"] = "HIGH"
            result["reason"] = "거래 통제 함수 없음"

        return result

    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        result["result"] = "UNKNOWN"
        result["confidence"] = "LOW"
        result["reason"] = f"예상치 못한 오류 발생: {str(e)}"
        return result
