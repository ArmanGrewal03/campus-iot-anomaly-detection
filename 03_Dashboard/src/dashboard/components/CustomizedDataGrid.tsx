import * as React from 'react';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';

const GATEWAY_BASE = 'http://127.0.0.1:8003';
const USER_SERVICE_BASE = `${GATEWAY_BASE}`;

interface NetworkLogRow {
  id: number;
  network_id: string;
  timestamp: string;
  os: string | null;
  browser: string | null;
  is_active: boolean;
}

const columns: GridColDef<NetworkLogRow>[] = [
  { field: 'id', headerName: 'ID', width: 80 },
  { field: 'network_id', headerName: 'Network ID', flex: 1, minWidth: 180 },
  {
    field: 'timestamp',
    headerName: 'Timestamp (UTC)',
    flex: 1,
    minWidth: 200,
    valueFormatter: (params) => {
      const value = params.value as string | null | undefined;
      if (!value) return '';
      try {
        return new Date(value).toLocaleString('en-US', { timeZone: 'UTC' });
      } catch {
        return value;
      }
    },
  },
  { field: 'os', headerName: 'OS', width: 120 },
  { field: 'browser', headerName: 'Browser', width: 140 },
  {
    field: 'is_active',
    headerName: 'Active Session',
    width: 140,
    type: 'boolean',
  },
];

export interface CustomizedDataGridProps {
  onInitialLoadComplete?: () => void;
  hideLoadingDuringInitialLoad?: boolean;
  /** When true, root uses height 100% to fill parent (e.g. match adjacent column). */
  fillHeight?: boolean;
}

export default function CustomizedDataGrid(props: CustomizedDataGridProps = {}) {
  const { onInitialLoadComplete, hideLoadingDuringInitialLoad, fillHeight } = props;
  const [rows, setRows] = React.useState<NetworkLogRow[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchLogs = async () => {
      try {
        setLoading(true);
        setError(null);

        // Get latest 100 network logs with predictions (if available)
        const res = await fetch(`${USER_SERVICE_BASE}/network-logs?limit=100&offset=0`);
        const json = await res.json() as {
          status?: string;
          logs?: Array<{
            id: number;
            network_id: string;
            timestamp: string;
            os: string | null;
            browser: string | null;
            is_active: boolean;
          }>;
          total?: number;
          detail?: string;
        };

        if (!res.ok || json.status !== 'success') {
          setRows([]);
          setError(null);
          return;
        }

        const logs = json.logs ?? [];
        const mapped: NetworkLogRow[] = logs.map((log) => ({
          id: log.id,
          network_id: log.network_id,
          timestamp: log.timestamp,
          os: log.os ?? null,
          browser: log.browser ?? null,
          is_active: !!log.is_active,
        }));

        setRows(mapped);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch network logs:', err);
        setRows([]);
        setError(null);
      } finally {
        setLoading(false);
        onInitialLoadComplete?.();
      }
    };

    fetchLogs();
  }, [onInitialLoadComplete]);

  return (
    <Box
      sx={{
        width: '100%',
        height: fillHeight ? '100%' : 400,
        minHeight: fillHeight ? 400 : undefined,
        ...(fillHeight ? { display: 'flex', flexDirection: 'column' } : {}),
      }}
    >
      <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ mb: 1, flexShrink: 0 }}>
        {loading && (
          <Stack direction="row" spacing={1} alignItems="center">
            <CircularProgress size={14} />
            <Typography variant="caption" color="text.secondary">
              Loading…
            </Typography>
          </Stack>
        )}
      </Stack>
      {error && (
        <Alert severity="warning" sx={{ mb: 1, flexShrink: 0 }}>
          {error}
        </Alert>
      )}
      <Box sx={fillHeight ? { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' } : undefined}>
      <DataGrid
        rows={rows}
        columns={columns}
        sx={fillHeight ? { height: '100%', minHeight: 400 } : undefined}
        getRowClassName={(params) =>
          params.indexRelativeToCurrentPage % 2 === 0 ? 'even' : 'odd'
        }
        initialState={{
          pagination: { paginationModel: { pageSize: 20 } },
        }}
        pageSizeOptions={[10, 20, 50]}
        disableColumnResize
        density="compact"
        loading={loading}
        slots={{
          noRowsOverlay: () => (
            <Stack alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
              <Typography variant="body2" color="text.secondary">
                No recent activity found.
              </Typography>
            </Stack>
          ),
        }}
        slotProps={{
          filterPanel: {
            filterFormProps: {
              logicOperatorInputProps: {
                variant: 'outlined',
                size: 'small',
              },
              columnInputProps: {
                variant: 'outlined',
                size: 'small',
                sx: { mt: 'auto' },
              },
              operatorInputProps: {
                variant: 'outlined',
                size: 'small',
                sx: { mt: 'auto' },
              },
              valueInputProps: {
                InputComponentProps: {
                  variant: 'outlined',
                  size: 'small',
                },
              },
            },
          },
        }}
      />
      </Box>
    </Box>
  );
}
