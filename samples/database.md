# Database migrations

Schema changes ship as SQL files under `db/migrations`.
The runner stores applied versions in the `schema_migrations` table.
Each version runs inside one transaction.
Failed SQL stops the runner so later versions do not apply.
