import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';

export default function CustomizedTreeView() {
  return (
    <Card
      variant="outlined"
      sx={{ display: 'flex', flexDirection: 'column', gap: '8px', flexGrow: 1 }}
    >
      <CardContent>
        <Typography component="h2" variant="subtitle2">
          Product tree
        </Typography>
        <SimpleTreeView
          aria-label="pages"
          multiSelect
          defaultExpandedItems={['1', '1.1']}
          defaultSelectedItems={['1.1', '1.1.1']}
          sx={{
            m: '0 -8px',
            pb: '8px',
            height: 'fit-content',
            flexGrow: 1,
            overflowY: 'auto',
          }}
        >
          <TreeItem itemId="1" label="Website">
            <TreeItem itemId="1.1" label="Home" />
            <TreeItem itemId="1.2" label="Pricing" />
            <TreeItem itemId="1.3" label="About us" />
            <TreeItem itemId="1.4" label="Blog">
              <TreeItem itemId="1.1.1" label="Announcements" />
              <TreeItem itemId="1.1.2" label="April lookahead" />
              <TreeItem itemId="1.1.3" label="What's new" />
              <TreeItem itemId="1.1.4" label="Meet the team" />
            </TreeItem>
          </TreeItem>
          <TreeItem itemId="2" label="Store">
            <TreeItem itemId="2.1" label="All products" />
            <TreeItem itemId="2.2" label="Categories">
              <TreeItem itemId="2.2.1" label="Gadgets" />
              <TreeItem itemId="2.2.2" label="Phones" />
              <TreeItem itemId="2.2.3" label="Wearables" />
            </TreeItem>
            <TreeItem itemId="2.3" label="Bestsellers" />
            <TreeItem itemId="2.4" label="Sales" />
          </TreeItem>
          <TreeItem itemId="4" label="Contact" />
          <TreeItem itemId="5" label="Help" />
        </SimpleTreeView>
      </CardContent>
    </Card>
  );
}
