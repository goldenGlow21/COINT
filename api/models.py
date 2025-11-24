"""
Database models for token scam detection system.
Designed based on the provided DB schema specification.
"""

from django.db import models
from django.utils import timezone
from decimal import Decimal


class TokenInfo(models.Model):
    """
    Core model storing token metadata and pair information.
    Acts as the central reference point for all analysis data.
    Primary key: token_addr_idx (auto-generated)
    """
    # token_addr_idx is auto-generated as primary key (id field)
    token_addr = models.CharField(
        max_length=42,
        unique=True,
        db_index=True,
        help_text="Actual token contract address"
    )
    pair_addr = models.CharField(
        max_length=42,
        help_text="Pair contract address"
    )
    token_create_ts = models.DateTimeField(
        help_text="Token creation timestamp"
    )
    lp_create_ts = models.DateTimeField(
        help_text="Liquidity pool creation timestamp"
    )
    pair_idx = models.IntegerField(
        help_text="Token index in pair (0 or 1)"
    )
    pair_type = models.CharField(
        max_length=50,
        help_text="Router type used by the pair"
    )
    token_creator_addr = models.CharField(
        max_length=42,
        blank=True,
        null=True,
        help_text="Token creator address"
    )

    class Meta:
        db_table = 'token_info'
        ordering = ['-id']

    def __str__(self):
        return f"Token {self.id}: {self.token_addr}"


class AnalysisJob(models.Model):
    """
    Tracks analysis job status and progress.
    Links to TokenInfo for job-token association.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('collecting_token', 'Collecting Token Info'),
        ('collecting_pair', 'Collecting Pair Events'),
        ('collecting_holder', 'Collecting Holder Info'),
        ('preprocessing', 'Preprocessing'),
        ('analyzing_honeypot_da', 'Honeypot Dynamic Analysis'),
        ('analyzing_honeypot_ml', 'Honeypot ML Analysis'),
        ('analyzing_exit_ml', 'Exit Scam ML Analysis'),
        ('aggregating', 'Aggregating Results'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    token_addr = models.CharField(
        max_length=42,
        db_index=True,
        help_text="Token address being analyzed"
    )
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='jobs',
        null=True,
        blank=True,
        help_text="Link to TokenInfo after collection"
    )

    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default='pending'
    )
    current_step = models.CharField(
        max_length=100,
        blank=True,
        help_text="Current processing step description"
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    error_message = models.TextField(blank=True, null=True)
    error_step = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        db_table = 'analysis_job'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['token_addr', '-created_at']),
            models.Index(fields=['status', '-created_at']),
        ]

    def __str__(self):
        return f"Job {self.id}: {self.token_addr} ({self.status})"


class Result(models.Model):
    """
    Final analysis result table.
    Stores aggregated scam detection results and risk scores.
    """
    # id field is auto-generated (separate from token_addr_idx)
    token_addr = models.CharField(
        max_length=42,
        unique=True,
        db_index=True,
        help_text="Analyzed token contract address"
    )

    risk_score = models.FloatField(
        help_text="Final scam score calculated after analysis"
    )
    scam_types = models.CharField(
        max_length=100,
        help_text="Detected scam categories"
    )
    victim_insights = models.TextField(
        help_text="Scam indicators found in analysis"
    )

    created_at = models.DateTimeField(
        default=timezone.now,
        help_text="Analysis completion timestamp"
    )

    class Meta:
        db_table = 'result'
        ordering = ['-created_at']

    def __str__(self):
        return f"Result {self.id}: {self.token_addr} (score: {self.risk_score})"


class PairEvent(models.Model):
    """
    Stores raw pair event data collected from blockchain.
    Contains Mint, Burn, Sync, Swap events from the pair contract.
    """
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='pair_events',
        db_column='token_addr_idx'
    )

    timestamp = models.DateTimeField(
        db_index=True,
        help_text="Event timestamp (ISO format)"
    )
    block_number = models.IntegerField(
        help_text="Block number of the transaction"
    )
    tx_hash = models.CharField(
        max_length=66,
        help_text="Transaction hash"
    )
    tx_from = models.CharField(
        max_length=42,
        help_text="Transaction sender address"
    )
    tx_to = models.CharField(
        max_length=42,
        help_text="Transaction recipient address"
    )
    evt_idx = models.IntegerField(
        help_text="Event log index in transaction"
    )
    evt_type = models.CharField(
        max_length=20,
        help_text="Event type (Mint, Burn, Sync, Swap, etc.)"
    )
    evt_log = models.JSONField(
        help_text="Preprocessed event log in JSON format"
    )

    lp_total_supply = models.DecimalField(
        max_digits=78,
        decimal_places=18,
        help_text="Current LP total supply"
    )

    class Meta:
        db_table = 'pair_evt'
        ordering = ['timestamp', 'evt_idx']
        indexes = [
            models.Index(fields=['token_info', 'timestamp']),
            models.Index(fields=['block_number']),
        ]

    def __str__(self):
        return f"{self.evt_type} at {self.timestamp}"


class HolderInfo(models.Model):
    """
    Stores token holder information.
    Each row represents a single holder's balance data.
    """
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='holders',
        db_column='token_addr_idx'
    )

    holder_addr = models.CharField(
        max_length=42,
        help_text="Holder wallet address"
    )
    balance = models.DecimalField(
        max_digits=78,
        decimal_places=18,
        help_text="Token balance held by this address"
    )
    rel_to_total = models.CharField(
        max_length=20,
        help_text="Percentage of total supply held"
    )

    class Meta:
        db_table = 'holder_info'
        ordering = ['-balance']
        indexes = [
            models.Index(fields=['token_info', '-balance']),
        ]

    def __str__(self):
        return f"Holder {self.holder_addr}: {self.rel_to_total}"


class HoneypotProcessedData(models.Model):
    """
    Preprocessed data for honeypot detection analysis.
    One record per token after preprocessing stage.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='honeypot_processed',
        primary_key=True,
        db_column='token_addr_idx'
    )

    # Trade counts
    total_buy_cnt = models.IntegerField()
    total_sell_cnt = models.IntegerField()
    total_owner_sell_cnt = models.IntegerField()
    total_non_owner_sell_cnt = models.IntegerField()

    # Imbalance metrics
    imbalance_rate = models.FloatField()
    total_windows = models.IntegerField()
    windows_with_activity = models.IntegerField()

    # Event counts
    total_burn_events = models.IntegerField()
    total_mint_events = models.IntegerField()
    s_owner_count = models.IntegerField()

    # Volume metrics
    total_sell_vol = models.DecimalField(max_digits=78, decimal_places=18)
    total_buy_vol = models.DecimalField(max_digits=78, decimal_places=18)
    total_owner_sell_vol = models.DecimalField(max_digits=78, decimal_places=18)

    # Log-transformed volumes
    total_sell_vol_log = models.FloatField()
    total_buy_vol_log = models.FloatField()
    total_owner_sell_vol_log = models.FloatField()

    # Additional metrics
    liquidity_event_mask = models.IntegerField()
    max_sell_share = models.FloatField()
    unique_sellers = models.IntegerField()
    unique_buyers = models.IntegerField()
    consecutive_sell_block_windows = models.IntegerField()
    total_sell_block_windows = models.IntegerField()

    class Meta:
        db_table = 'honeypot_processed_data'

    def __str__(self):
        return f"Honeypot data for token {self.token_info.id}"


class ExitProcessedData(models.Model):
    """
    Preprocessed data for exit scam detection analysis.
    Multiple records per token (one per 5-second window).
    """
    win_id = models.IntegerField(
        help_text="Window identifier (per token)"
    )
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='exit_processed',
        db_column='token_addr_idx'
    )

    # Window metadata
    win_start_ts = models.DateTimeField()
    win_start_block = models.IntegerField()
    win_tx_count = models.IntegerField()
    win_blocks = models.IntegerField()

    # 5-second window LP metrics
    lp_start_5s = models.DecimalField(max_digits=78, decimal_places=18)
    lp_end_5s = models.DecimalField(max_digits=78, decimal_places=18)
    lp_drop_amount_5s = models.DecimalField(max_digits=78, decimal_places=18)
    burn_frac_5s = models.FloatField()

    # 5-second window reserve metrics
    reserve_token_start_5s = models.DecimalField(max_digits=78, decimal_places=18)
    reserve_token_end_5s = models.DecimalField(max_digits=78, decimal_places=18)
    reserve_token_drop_frac_5s = models.FloatField()

    # 5-second window event metrics
    lp_mint_amount_5s = models.DecimalField(max_digits=78, decimal_places=18)
    lp_burn_amount_5s = models.DecimalField(max_digits=78, decimal_places=18)
    mint_events_5s = models.IntegerField()
    burn_events_5s = models.IntegerField()
    swap_events_5s = models.IntegerField()
    time_since_last_mint_sec_5s = models.FloatField()

    # 5-second window LP peak metrics
    lp_peak_drop_frac_5s = models.FloatField()
    lp_start_peak_frac_5s = models.FloatField()

    # 5-second window swap volumes
    swap_base_sell_volume_5s = models.DecimalField(max_digits=78, decimal_places=18)
    swap_base_buy_volume_5s = models.DecimalField(max_digits=78, decimal_places=18)
    swap_quote_sell_volume_5s = models.DecimalField(max_digits=78, decimal_places=18)
    swap_quote_buy_volume_5s = models.DecimalField(max_digits=78, decimal_places=18)

    # 5-second window cumulative metrics
    cum_base_minted_5s = models.DecimalField(max_digits=78, decimal_places=18)
    cum_base_burned_5s = models.DecimalField(max_digits=78, decimal_places=18)
    cum_quote_minted_5s = models.DecimalField(max_digits=78, decimal_places=18)
    cum_quote_burned_5s = models.DecimalField(max_digits=78, decimal_places=18)

    # Holder concentration
    holder_top1_supply_pct = models.FloatField()
    holder_pair_supply_pct = models.FloatField()
    holder_top20_supply_pct = models.FloatField()

    # 60-second rolling window metrics
    mint_sell_swap_count_60s = models.IntegerField()
    mint_sell_base_volume_60s = models.DecimalField(max_digits=78, decimal_places=18)
    mint_sell_base_abs_max_60s = models.DecimalField(max_digits=78, decimal_places=18)
    mint_sell_to_reserve_max_ratio_60s = models.FloatField()
    mint_sell_to_reserve_avg_ratio_60s = models.FloatField()
    mint_sell_quote_volume_60s = models.DecimalField(max_digits=78, decimal_places=18)

    # 600-second rolling window event counts
    event_count_600s = models.IntegerField()
    burn_events_600s = models.IntegerField()
    mint_events_600s = models.IntegerField()
    swap_events_600s = models.IntegerField()

    # 600-second rolling window LP metrics
    lp_start_600s = models.DecimalField(max_digits=78, decimal_places=18)
    lp_drop_frac_600s = models.FloatField()
    lp_cum_drawdown_600s = models.FloatField()
    lp_burn_amount_600s = models.DecimalField(max_digits=78, decimal_places=18)
    lp_mint_amount_600s = models.DecimalField(max_digits=78, decimal_places=18)
    time_since_last_mint_sec_600s = models.FloatField()
    consecutive_drop_windows_600s = models.IntegerField()

    # 600-second rolling window reserve and price metrics
    reserve_token_start_600s = models.DecimalField(max_digits=78, decimal_places=18)
    price_ratio_start_600s = models.FloatField()
    price_ratio_end_600s = models.FloatField()

    class Meta:
        db_table = 'exit_processed_data'
        ordering = ['token_info', 'win_id']
        indexes = [
            models.Index(fields=['token_info', 'win_id']),
            models.Index(fields=['win_start_ts']),
        ]
        unique_together = [['token_info', 'win_id']]

    def __str__(self):
        return f"Exit data for token {self.token_info.id}, window {self.win_id}"


class HoneypotDaResult(models.Model):
    """
    Stores honeypot dynamic analysis results.
    One record per token, created by honeypot_DA module.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='honeypot_da_result',
        primary_key=True,
        db_column='token_addr_idx'
    )

    verified = models.BooleanField(
        help_text="Contract verified on Etherscan"
    )

    # Buy/Sell test
    buy_sell_result = models.BooleanField(
        help_text="Buy and sell both successful"
    )
    buy_sell_return_rate = models.FloatField(
        null=True,
        blank=True,
        help_text="Return rate percentage (0-100)"
    )

    # Blacklist check
    blacklist_result = models.BooleanField()
    blacklist_confidence = models.CharField(max_length=10)

    # Trading suspend check
    trading_suspend_result = models.BooleanField()
    trading_suspend_confidence = models.CharField(max_length=10)

    # Exterior call check
    exterior_call_result = models.BooleanField()
    exterior_call_confidence = models.CharField(max_length=10)

    # Unlimited mint
    unlimited_mint_result = models.BooleanField()
    unlimited_mint_confidence = models.CharField(max_length=10)

    # Balance manipulation
    balance_manipulation_result = models.BooleanField()
    balance_manipulation_confidence = models.CharField(max_length=10)

    # Tax manipulation
    tax_manipulation_result = models.BooleanField()
    tax_manipulation_confidence = models.CharField(max_length=10)

    # Existing holders check
    existing_holders_result = models.BooleanField()
    existing_holders_confidence = models.CharField(max_length=10)

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'honeypot_da_result'

    def __str__(self):
        return f"Honeypot DA result for token {self.token_info.id}"


class HoneypotMlResult(models.Model):
    """
    Stores honeypot ML analysis results.
    One record per token, created by honeypot_ML module.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='honeypot_ml_result',
        primary_key=True,
        db_column='token_addr_idx'
    )

    is_honeypot = models.BooleanField(
        help_text="Honeypot prediction result"
    )
    probability = models.FloatField(
        help_text="Honeypot probability (0-1)"
    )
    risk_level = models.CharField(
        max_length=20,
        help_text="Risk level: CRITICAL/HIGH/MEDIUM/LOW/VERY_LOW"
    )
    threshold = models.FloatField(
        help_text="Decision threshold from model"
    )
    top_contributing_features = models.JSONField(
        help_text="Top 5 contributing features with values and directions"
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'honeypot_ml_result'

    def __str__(self):
        return f"Honeypot ML result for token {self.token_info.id} (prob: {self.probability:.4f})"


class ExitMlResult(models.Model):
    """
    Stores exit scam ML analysis results.
    One record per token, created by exit_ML module.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='exit_ml_result',
        primary_key=True,
        db_column='token_addr_idx'
    )

    probability = models.FloatField(
        help_text="Exit scam probability (0-1)"
    )
    threshold = models.FloatField(
        default=0.6047,
        help_text="Decision threshold from model"
    )
    is_exit_scam = models.BooleanField(
        help_text="True if probability >= threshold"
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'exit_ml_result'

    def __str__(self):
        return f"Exit ML result for token {self.token_info.id} (prob: {self.probability:.4f})"


class ExitMlDetectTransaction(models.Model):
    """
    Stores top suspicious transactions detected by exit_ML model.
    Up to 3 records per ExitMlResult (detect_tx_1, detect_tx_2, detect_tx_3).
    """
    exit_ml_result = models.ForeignKey(
        ExitMlResult,
        on_delete=models.CASCADE,
        related_name='detect_transactions',
        db_column='exit_ml_result_id'
    )

    rank = models.IntegerField(
        help_text="Transaction rank (1, 2, or 3)"
    )
    timestamp = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Transaction timestamp"
    )
    tx_hash = models.CharField(
        max_length=66,
        null=True,
        blank=True,
        help_text="Transaction hash"
    )
    feature_values = models.JSONField(
        help_text="Top instance feature values (scaled) as JSON"
    )

    class Meta:
        db_table = 'exit_ml_detect_transaction'
        ordering = ['exit_ml_result', 'rank']
        unique_together = [['exit_ml_result', 'rank']]

    def __str__(self):
        return f"Detect TX {self.rank} for result {self.exit_ml_result_id}"


class ExitMlDetectStatic(models.Model):
    """
    Stores static (bag-level) features detected by exit_ML model.
    One record per ExitMlResult.
    """
    exit_ml_result = models.OneToOneField(
        ExitMlResult,
        on_delete=models.CASCADE,
        related_name='detect_static',
        primary_key=True,
        db_column='exit_ml_result_id'
    )

    feature_values = models.JSONField(
        help_text="Top static feature values as JSON"
    )

    class Meta:
        db_table = 'exit_ml_detect_static'

    def __str__(self):
        return f"Detect static for result {self.exit_ml_result_id}"
