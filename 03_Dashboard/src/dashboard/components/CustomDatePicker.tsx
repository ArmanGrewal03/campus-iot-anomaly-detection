import * as React from 'react';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import CalendarMonthRoundedIcon from '@mui/icons-material/CalendarMonthRounded';

export default function CustomDatePicker() {
  const [today, setToday] = React.useState<Date>(new Date());

  React.useEffect(() => {
    const interval = setInterval(() => {
      setToday(new Date());
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  const formatted = React.useMemo(
    () => today.toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' }),
    [today],
  );

  return (
    <Stack
      direction="row"
      spacing={0.75}
      alignItems="center"
      sx={{
        minWidth: 210,
        px: 1.25,
        py: 0.75,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        bgcolor: 'background.paper',
      }}
      aria-label="Current date"
    >
      <CalendarMonthRoundedIcon fontSize="small" sx={{ color: 'text.secondary' }} />
      <Typography variant="body2" sx={{ whiteSpace: 'nowrap' }}>
        {formatted}
      </Typography>
    </Stack>
  );
}
