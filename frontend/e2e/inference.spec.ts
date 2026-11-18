import { expect, test, type Page } from '@playwright/test';

async function openPhoto(page: Page) {
  await page.goto('/');
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 32;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas unavailable');
    ctx.fillStyle = '#788595';
    ctx.fillRect(0, 0, 32, 32);
    return canvas.toDataURL('image/png').split(',')[1] ?? '';
  });
  await page
    .locator('input[type=file]')
    .first()
    .setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: Buffer.from(png, 'base64'),
    });
}

test('uses the backend default and describes unavailable models', async ({ page }) => {
  await page.route('**/api/models', (route) =>
    route.fulfill({
      json: {
        default_geometry: 'depth-pro',
        default_normal: 'marigold-normals',
        unavailable: {
          'moge-3': 'MoGe-3 requires Linux or Windows.',
          'marigold-v2': 'Marigold V2 requires a Linux CUDA backend.',
        },
      },
    }),
  );
  await openPhoto(page);
  const selector = page.getByLabel('Geometry Model');
  await expect(selector).toHaveValue('depth-pro');
  await expect(selector.locator('option[value="moge-3"]')).toBeDisabled();
  const refineCheckbox = page.getByRole('checkbox', { name: 'Refine depth with normals' });
  await expect(refineCheckbox).toBeEnabled();
  await refineCheckbox.check();
  const normalSelector = page.getByLabel('Normal Model');
  await expect(normalSelector).toHaveValue('marigold-normals');
  await expect(normalSelector.locator('option[value="marigold-v2"]')).toBeDisabled();
  await page.getByText('Unavailable models', { exact: true }).first().click();
  await expect(page.getByText('MoGe-3 requires Linux or Windows.', { exact: false })).toBeVisible();
});

test('labels model usage and shows the V2 download size before inference', async ({ page }) => {
  await page.route('**/api/models', (route) =>
    route.fulfill({ json: { default_geometry: 'depth-pro', unavailable: {} } }),
  );
  await openPhoto(page);
  const appearance = page.getByLabel('Appearance Model');
  await expect(appearance).toHaveAccessibleDescription(/Commercial use allowed with Open RAIL/);
  await expect(appearance.locator('option[value="latent-intrinsics"]')).toHaveText(
    /non-commercial/,
  );
  await expect(appearance.locator('option[value="iid-in-the-wild"]')).toHaveText(/non-commercial/);
  await appearance.selectOption('iid-in-the-wild');
  await expect(appearance).toHaveAccessibleDescription('Non-commercial (academic use only)');
  await appearance.selectOption('latent-intrinsics');
  await expect(appearance).toHaveAccessibleDescription('Non-commercial');
  await appearance.selectOption('marigold-v2');
  await expect(appearance).toHaveAccessibleDescription('Commercial use allowed');
  await expect(page.getByText(/first run downloads up to about 43 GB/)).toBeVisible();
});

test('compares scale-aligned runs before changing the active reconstruction', async ({ page }) => {
  await page.setViewportSize({ width: 1160, height: 800 });
  await page.route('**/api/models', (route) =>
    route.fulfill({ json: { default_geometry: 'moge-3', unavailable: {} } }),
  );
  const requests: URLSearchParams[] = [];
  await page.route('**/api/geometry/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/infer')) {
      const model = url.searchParams.get('model') ?? 'moge-3';
      requests.push(url.searchParams);
      const refined = url.searchParams.get('refine_normals') === 'true';
      if (model === 'depth-pro' && !refined) {
        await route.fulfill({
          status: 503,
          json: { detail: 'Depth Pro unavailable for this run.' },
        });
        return;
      }
      await route.fulfill({
        json: {
          id: refined ? `${model}-refined` : model,
          model,
          model_id: 'test/' + model,
          image_sha256: 'same-image',
          inference_width: 2,
          inference_height: 2,
          normal_model_id: refined ? 'test/normals' : null,
          intrinsics: [
            [1, 0, 0.5],
            [0, 1, 0.5],
            [0, 0, 1],
          ],
          depth_range: [1, 2],
          valid_pixel_count: 4,
          cached: false,
          has_normals: false,
        },
      });
    } else if (url.pathname.endsWith('/maps/mask')) {
      await route.fulfill({ body: Buffer.from([1, 1, 1, 1]) });
    } else {
      const data = url.pathname.endsWith('/maps/depth')
        ? url.pathname.includes('depth-pro')
          ? [1, 1, 2, 1]
          : [2, 2, 2, 2]
        : [-1, -1, 2, 1, -1, 2, -1, 1, 2, 1, 1, 2];
      await route.fulfill({ body: Buffer.from(new Float32Array(data).buffer) });
    }
  });
  await openPhoto(page);
  const run = page.getByRole('button', { name: 'Run Geometry Model' });
  await run.click();
  await expect(page.locator('[title="test/moge-3"]')).toHaveText('MoGe-3');
  await page.getByLabel('Geometry Model').selectOption('depth-pro');
  await run.click();
  await expect(
    page.getByRole('alert').filter({ hasText: 'Depth Pro unavailable for this run.' }),
  ).toBeVisible();
  await expect(page.locator('[title="test/moge-3"]')).toHaveText('MoGe-3');
  await page.getByRole('checkbox', { name: 'Refine depth with normals' }).check();
  await expect(page.getByLabel('Normal Model')).toHaveValue('marigold-normals');
  await page.getByRole('checkbox', { name: 'Smooth depth & normals' }).check();
  await run.click();
  await expect(page.locator('[title="test/moge-3"]')).toHaveText('MoGe-3');
  await expect(page.getByText('2.00×', { exact: true })).toBeVisible();
  await expect(page.getByText('25.0%', { exact: true })).toBeVisible();
  const preview = page.getByRole('button', { name: 'View Depth Difference' });
  await preview.click();
  await expect(preview).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByLabel('Aligned depth ratio for compared model')).toBeVisible();
  const workspace = page.locator('.editor-workspace');
  const bounds = await workspace.boundingBox();
  expect(bounds?.x).toBe(0);
  expect(bounds?.width).toBe(1160);
  await page.getByRole('button', { name: 'Use Compared Geometry' }).click();
  await expect(page.locator('[title="test/depth-pro"]')).toHaveText('Depth Pro + normals');
  expect(requests[2]?.get('refine_normals')).toBe('true');
  expect(requests[2]?.get('normal_model')).toBe('marigold-normals');
  expect(requests[2]?.get('smooth_iterations')).toBe('3');
});

test('enables texture for cached albedo and keeps it if a later run fails', async ({ page }) => {
  await page.route('**/api/models', (route) =>
    route.fulfill({ json: { default_geometry: 'depth-pro', unavailable: {} } }),
  );
  await page.route('**/api/appearance/infer?**', (route) => {
    if (new URL(route.request().url()).searchParams.get('model') !== 'latent-intrinsics') {
      return route.fulfill({ status: 503, json: { detail: 'Appearance model unavailable.' } });
    }
    return route.fulfill({
      json: {
        id: 'albedo',
        model_id: 'test/latent-intrinsics',
        cached: true,
        appearance_width: 4,
        appearance_height: 4,
        prediction_space: 'linear',
        arrays: { albedo: { shape: [4, 4, 3], dtype: 'float32' } },
      },
    });
  });
  const maps: string[] = [];
  await page.route('**/api/appearance/albedo/maps/**', (route) => {
    const path = new URL(route.request().url()).pathname;
    maps.push(path);
    return route.fulfill({ body: Buffer.from(new Float32Array(48).fill(0.5).buffer) });
  });
  await openPhoto(page);
  await page.getByLabel('Appearance Model').selectOption('latent-intrinsics');
  const generate = page.getByRole('button', { name: 'Generate Appearance' }).first();
  await generate.click();
  await expect(page.getByText('Derived from albedo')).toBeVisible();
  await expect(
    page.getByRole('group', { name: 'View mode' }).getByRole('button', { name: 'Shading' }),
  ).toBeEnabled();
  const texture = page
    .getByRole('group', { name: 'Paint compositing mode' })
    .getByRole('button', { name: 'Texture' });
  await texture.click();
  await expect(texture).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Texture Detail', { exact: true })).toBeVisible();
  await page.getByLabel('Appearance Model').selectOption('marigold-iid');
  await generate.click();
  await expect(
    page.getByRole('alert').filter({ hasText: 'Appearance model unavailable.' }),
  ).toBeVisible();
  await expect(page.getByText('Derived from albedo')).toBeVisible();
  await expect(texture).toHaveAttribute('aria-pressed', 'true');
  expect(maps).toEqual(['/api/appearance/albedo/maps/albedo']);
});
