// Mints short-lived tokens so the browser can upload a PDF straight to Vercel Blob,
// bypassing the 4.5 MB request-body cap on functions. This is the one piece of the
// backend that is not Python: Vercel Blob's upload handshake must be signed with
// BLOB_READ_WRITE_TOKEN and only the JS SDK implements it. Python still does all
// the indexing — it reads the finished blob back over plain HTTPS.
import { handleUpload, del, type HandleUploadBody } from '@vercel/blob/client';

export const config = { runtime: 'nodejs' };

function authorised(request: Request): boolean {
  const expected = process.env.ADMIN_PASSWORD;
  const supplied = request.headers.get('x-admin-password');
  return Boolean(expected) && supplied === expected;
}

export default async function handler(request: Request): Promise<Response> {
  if (!authorised(request)) {
    return Response.json({ error: 'Incorrect password' }, { status: 401 });
  }

  // DELETE ?url=... removes a stored PDF, used when a document is deleted.
  if (request.method === 'DELETE') {
    const url = new URL(request.url).searchParams.get('url');
    if (!url) return Response.json({ error: 'Missing url' }, { status: 400 });
    await del(url);
    return Response.json({ ok: true });
  }

  try {
    const body = (await request.json()) as HandleUploadBody;
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: ['application/pdf'],
        maximumSizeInBytes: 50 * 1024 * 1024,   // Gemini's own per-PDF ceiling
        addRandomSuffix: false,
      }),
      // Indexing is kicked off by the browser once the upload resolves, so there
      // is nothing to do here; Vercel still requires the callback to exist.
      onUploadCompleted: async () => {},
    });
    return Response.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Upload failed';
    return Response.json({ error: message }, { status: 400 });
  }
}
