# API error handling

The HTTP client classifies responses before it returns them to callers.
Status codes 400 and 404 are permanent failures.
Status codes 429 and 500 through 599 are transient failures.

## Retry logic

The file that implements retry logic is `app/http/retry.py`.
`retry_request` applies exponential backoff with jitter and allows three attempts.
Retryable status codes live in `RETRYABLE_STATUS_CODES`.

## Timeouts

Connect and read timeouts are configured in `app/http/client.py`.
The timeout values are `connect_timeout` and `read_timeout`.
Pass `allow_repeat=False` to disable another attempt after a slow response.
